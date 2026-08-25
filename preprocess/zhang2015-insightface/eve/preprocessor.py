"""EVE 预处理（zhang2015-insightface 管线）

数据流（与 gazecapture 同构）：

    每被试（官方名 train01..39 / val01..05，val 为平台 test）:
        每刺激 step 目录 × 每相机 {basler, webcam_l, webcam_c, webcam_r}:
            {cam}.mp4 原始帧（1920x1080，官方已去畸变，BGR）
            ↓ 5Hz 采样：i % round(fps/target_hz) == 0（basler 每 12 帧、webcam 每 6 帧，
              四相机起始帧对齐 → 同一时刻四视角）
            ↓ 有效性门控：{cam}.h5 的 face_PoG_tobii/validity（无效 = no_annotation）
            ↓ insightface 106 关键点 → PnP 头姿（K 取自该 h5，dist=0——mp4 已去畸变）
            ↓ 视线标签: face_PoG_tobii 屏幕像素 → 屏幕系 mm（原点左上，+x 右，+y 下，z=0 屏幕面）
              → camera_transformation → 相机系 3D 点 (mm)
              （该链路已与官方 face_g_tobii 对拍至 1e-7 rad，见 dataset_report.md §4）
            ↓ normalizeData_face 归一化 → 224x224 face_patch + (theta, phi)
            → output_dir/<split>/<被试>.h5（face_patch 为 BGR 存储，与其他数据集一致）

原始帧回溯（h5 自描述，frame_index 单独无法定位）：
    attrs['steps'] / attrs['cameras'] 为有序名字列表，原始帧 =
    raw_data_dir/<被试名>/<steps[step_index]>/<cameras[cam_index]>.mp4 的第 frame_index 帧
    （frame_index 与 {cam}.h5 行号、{cam}.timestamps.txt 行号一致）

运行注意：需 conda activate opengaze（onnxruntime GPU 依赖 LD_LIBRARY_PATH，
否则静默回退 CPU，慢 ~10 倍；见 CLAUDE.md「预处理注意」）。
"""
import json
import queue
import threading
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from preprocess.common import FailureRecorder
from utils.logger import get_logger
from utils.normalization import estimateHeadPose, normalizeData_face, vector_to_angles

log = get_logger('preprocess.eve')

# ---- 管线常量（与 gazecapture / normalize_xgaze 一致）----
LANDMARK_USE = [20, 23, 26, 29, 15, 19]   # 3D face model 中双眼四角+两鼻角
IDX6 = [35, 39, 89, 93, 78, 84]           # insightface 106 点对应索引
FACE_MODEL_USE = None                     # 延迟到模块初始化（避免 import 期读盘）

FPS = {'basler': 60, 'webcam_l': 30, 'webcam_c': 30, 'webcam_r': 30}
DIST = np.zeros((4, 1))                   # mp4 已去畸变，PnP 畸变系数取零


def _screen_px_to_cam_mm(pog_px, T, mmp):
    """PoG 屏幕像素 → 相机系 3D 点 (mm)，dataset_report.md §4 定稿公式"""
    p = T @ np.array([pog_px[0] * mmp[0], pog_px[1] * mmp[1], 0.0, 1.0])
    return p[:3]


class EVEPreprocessor:
    """EVE 数据预处理器（Zhang2015 归一化 + insightface 关键点，4 相机 5Hz）"""

    def __init__(self, config):
        global FACE_MODEL_USE
        self.config = config
        if FACE_MODEL_USE is None:
            fm = np.loadtxt(Path(__file__).parent.parent / 'face_model_xgaze.txt')
            FACE_MODEL_USE = fm.reshape(50, 1, 3)[LANDMARK_USE, :]
        self.face_model_use = FACE_MODEL_USE
        self.cameras = list(config.cameras)
        self.subject_filter = set(config.subjects) if config.subjects else None
        self.split_subjects = {s: list(v) for s, v in vars(config.splits).items()}
        self.splits = list(vars(config.splits))
        self.intervals = {c: round(FPS[c] / config.target_hz) for c in self.cameras}

    # ------------------------------------------------------------ 单 step×cam
    def _plan(self, step_dir, cam, recorder, rec_subject):
        """读取一步一相机的标定与标签，确定候选帧。

        返回 dict(K/T/mmp/pog/cand) 或 None（该 step×cam 不可用，已记失败）。
        cand = 采样且 PoG 有效的帧号；采样但无效的候选帧记 no_annotation。
        """
        h5_path = step_dir / f'{cam}.h5'
        mp4_path = step_dir / f'{cam}.mp4'
        try:
            with h5py.File(h5_path, 'r') as f:
                if 'face_PoG_tobii/data' not in f:
                    recorder.add(rec_subject, '<step>', 'no_gt_field')
                    return None
                K = np.array(f['camera_matrix'], dtype=float)
                T = np.array(f['camera_transformation'], dtype=float)
                mmp = np.array(f['millimeters_per_pixel'], dtype=float)
                pog = np.array(f['face_PoG_tobii/data'])
                valid = np.array(f['face_PoG_tobii/validity'])
        except Exception as e:
            recorder.add(rec_subject, '<step>', f'error:{type(e).__name__}:{e}')
            return None
        if not mp4_path.is_file():
            recorder.add(rec_subject, '<step>', 'mp4_missing')
            return None

        n_frames = len(valid)
        if getattr(self.config, 'max_frames', 0):
            n_frames = min(n_frames, self.config.max_frames)   # 调试：流内前 N 帧
        sampled = np.arange(0, n_frames, self.intervals[cam])
        cand = sampled[valid[sampled]]
        for i in sampled[~valid[sampled]]:
            recorder.add(rec_subject, f'frame{i:05d}', 'no_annotation')
        return {'K': K, 'T': T, 'mmp': mmp, 'pog': pog, 'cand': cand.tolist(),
                'n_sampled': len(sampled)}

    # ------------------------------------------------------------- 读帧线程
    @staticmethod
    def _read_stream(mp4_path, cand, q):
        """顺序解码 mp4，仅把候选帧入队；异常入队由主线程处理"""
        try:
            cap = cv2.VideoCapture(str(mp4_path))
            cand_set = set(cand)
            got, i = 0, 0
            while got < len(cand):
                ok, img = cap.read()
                if not ok:
                    break
                if i in cand_set:
                    q.put((i, img))
                    got += 1
                i += 1
            cap.release()
            q.put(None)
        except Exception as e:   # noqa: BLE001
            q.put(e)

    # ------------------------------------------------------------- 单被试
    def process_subject(self, subject, split, app, recorder: FailureRecorder):
        subject_dir = Path(self.config.raw_data_dir) / subject
        steps = sorted(d.name for d in subject_dir.iterdir()
                       if d.is_dir() and d.name.startswith('step'))
        if getattr(self.config, 'max_steps', 0):
            steps = steps[:self.config.max_steps]              # 调试：前 N 步

        # 预读全部 step×cam 的计划（同时把无效候选记入 recorder）
        plans = []
        for si, step in enumerate(steps):
            for cam in self.cameras:
                plan = self._plan(subject_dir / step, cam, recorder,
                                  f'{subject}/{step}/{cam}')
                if plan is not None:
                    plans.append((si, cam, plan))
        n_cand_total = sum(p['n_sampled'] for *_, p in plans)
        n_max = sum(len(p['cand']) for *_, p in plans)
        out_path = Path(self.config.output_dir) / split / f'{subject}.h5'
        out_path.parent.mkdir(parents=True, exist_ok=True)

        h5 = h5py.File(out_path, 'w')
        h5.attrs['steps'] = json.dumps(steps)
        h5.attrs['cameras'] = json.dumps(self.cameras)
        h5.attrs['target_hz'] = self.config.target_hz
        h5.attrs['source'] = str(subject_dir)
        h5.attrs['trace'] = ('raw frame = <source>/<steps[step_index]>/'
                             '<cameras[cam_index]>.mp4 #frame_index')
        dsets = {
            'frame_index': h5.create_dataset('frame_index', (n_max, 1), np.int32,
                                             chunks=(1, 1), maxshape=(None, 1)),
            'cam_index': h5.create_dataset('cam_index', (n_max, 1), np.int32,
                                           chunks=(1, 1), maxshape=(None, 1)),
            'step_index': h5.create_dataset('step_index', (n_max, 1), np.int32,
                                            chunks=(1, 1), maxshape=(None, 1)),
            'face_patch': h5.create_dataset('face_patch', (n_max, 224, 224, 3), np.uint8,
                                            chunks=(1, 224, 224, 3),
                                            maxshape=(None, 224, 224, 3),
                                            compression='lzf'),
            'face_mat_norm': h5.create_dataset('face_mat_norm', (n_max, 3, 3), float,
                                               chunks=(1, 3, 3), maxshape=(None, 3, 3)),
            'facial_landmarks_2d': h5.create_dataset(
                'facial_landmarks_2d', (n_max, 106, 2), np.float32,
                chunks=(1, 106, 2), maxshape=(None, 106, 2)),
            'face_gaze': h5.create_dataset('face_gaze', (n_max, 2), float,
                                           chunks=(1, 2), maxshape=(None, 2)),
        }

        written = 0
        t_sub = time.time()
        pbar = tqdm(total=n_cand_total, desc=f'{subject}({split})', unit='帧',
                    ncols=100, leave=False)
        try:
            for si, cam, plan in plans:
                ci = self.cameras.index(cam)
                q = queue.Queue(maxsize=8)
                threading.Thread(
                    target=self._read_stream,
                    args=(subject_dir / steps[si] / f'{cam}.mp4', plan['cand'], q),
                    daemon=True).start()
                received = 0
                while True:
                    item = q.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item
                    i, img = item
                    received += 1
                    rec_sample = f'frame{i:05d}'
                    if img is None:
                        recorder.add(f'{subject}/{steps[si]}/{cam}', rec_sample,
                                     'decode_failed')
                        pbar.update(1)
                        continue
                    faces = app.get(img)
                    if len(faces) == 0:
                        recorder.add(f'{subject}/{steps[si]}/{cam}', rec_sample,
                                     'no_face_detected')
                        pbar.update(1)
                        continue

                    lm106 = faces[0].landmark_2d_106
                    pts2d = lm106[IDX6].reshape(6, 1, 2).astype(float)
                    rvec, tvec = estimateHeadPose(
                        pts2d, self.face_model_use, plan['K'], DIST)
                    gaze_point = _screen_px_to_cam_mm(
                        plan['pog'][i], plan['T'], plan['mmp'])
                    img_warped, hr_norm, gc_normalized = normalizeData_face(
                        img, self.face_model_use, rvec, tvec, gaze_point,
                        plan['K'])[:3]
                    R = cv2.Rodrigues(hr_norm)[0] @ cv2.Rodrigues(rvec)[0].T
                    g_theta, g_phi = vector_to_angles(gc_normalized.flatten())

                    dsets['frame_index'][written] = i
                    dsets['cam_index'][written] = ci
                    dsets['step_index'][written] = si
                    dsets['face_patch'][written] = img_warped
                    dsets['face_mat_norm'][written] = R
                    dsets['facial_landmarks_2d'][written] = lm106
                    dsets['face_gaze'][written] = (g_theta, g_phi)
                    written += 1
                    pbar.update(1)
                    pbar.set_postfix({'入库': written})
                for i in plan['cand'][received:]:   # 流提前结束的候选帧
                    recorder.add(f'{subject}/{steps[si]}/{cam}', f'frame{i:05d}',
                                 'decode_failed')
                    pbar.update(1)
        finally:
            pbar.close()
            for d in dsets.values():
                d.resize((written,) + d.shape[1:])
            h5.close()

        log.info(f'{subject}({split}): {written}/{n_cand_total} 候选入库 '
                 f'({len(plans)} step×cam) -> {out_path.name} '
                 f'({(time.time() - t_sub) / 60:.1f} min)')
        return written

    # ------------------------------------------------------------------ 入口
    def run(self, recorder: FailureRecorder):
        from insightface.app import FaceAnalysis
        app = FaceAnalysis('buffalo_l',
                           allowed_modules=['detection', 'landmark_2d_106'],
                           providers=self.config.providers)
        app.prepare(ctx_id=0)

        total_n = 0
        t_all = time.time()
        for split in self.splits:
            subjects = self.split_subjects[split]
            if self.subject_filter:
                subjects = [s for s in subjects if s in self.subject_filter]
            log.info(f'===== split {split}: {len(subjects)} 个被试，'
                     f'输出 {Path(self.config.output_dir) / split} '
                     f'(采样间隔 {self.intervals}) =====')
            overall = tqdm(subjects, desc=f'eve-{split}', unit='被试',
                           ncols=100, leave=True)
            for subject in overall:
                try:
                    total_n += self.process_subject(subject, split, app, recorder)
                except Exception as e:  # 单被试失败不中断整体
                    log.error(f'[错误] {subject}: {type(e).__name__}: {e}')
                    recorder.add(subject, '<subject>',
                                 f'error:{type(e).__name__}:{e}')
                overall.set_postfix({'累计样本': total_n})
            overall.close()
        log.info(f'全部完成: 共 {total_n} 样本, '
                 f'总耗时 {(time.time() - t_all) / 3600:.2f} 小时')
        return total_n


def run(config, recorder):
    """preprocess.py 入口适配"""
    return EVEPreprocessor(config).run(recorder)
