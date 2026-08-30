"""ours-without-roll 版 XGaze 归一化（第三版，2026-08-29）

在 zhang2015-specific-face-model（v2，逐被试逐相机个性化模型）基础上的归一化改进：
  - normalizeData_face(fixed_forward=True)：虚拟相机光轴固定为原相机 z 轴 [0,0,1]，
    down 仍由 hRx 确定（横滚随头稳定），主点平移偏移使人脸中心落在归一化图像中心；
    归一化头姿不再被 face_center 方向估计影响，图像保留头姿 pitch/yaw 外观
  - 新增 h5 字段 face_gaze_head (N,2)：HCS（头部坐标系）视线 (pitch,yaw) 弧度，
    gc_head = hR_norm^T @ gc_normalized = hR^T @ (gaze_point − face_center)，
    对虚拟相机选择不变（CCS 视线与 HCS 视线一一对应，互可换算）
  - 个性化模型直接复用 v2 建模产物（ylx 盘 face_models，不重建），
    建模未覆盖的相机回退通用模型并记 fallback_generic

h5 字段：frame_index | cam_index | face_patch (224,224,3) | face_mat_norm |
         facial_landmarks_2d | face_gaze (CCS) | face_gaze_head (HCS)
用法：python preprocess.py --dataset xgaze --method ours-without-roll
      调试：--set subjects=[0] max_frames=2 output_dir=/tmp/owr_test
"""
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from preprocess.common import FailureRecorder
from utils.logger import get_logger
from utils.normalization import estimateHeadPose, normalizeData_face, vector_to_angles

log = get_logger('preprocess.ours_without_roll.xgaze')

RIGID = [i for i in list(range(33, 43)) + list(range(72, 87)) + list(range(87, 97))
         if i not in (34, 38, 86, 88, 92, 94, 95)]


class OursWithRollXGazePreprocessor:
    """XGaze 归一化：个性化模型 PnP + fixed_forward 相机系归一化 + HCS 视线标注"""

    IDX6 = [35, 39, 89, 93, 78, 84]
    FLIP_CAMERAS = [3, 6, 13]
    NUM_CAMS = 18

    def __init__(self, config):
        self.config = config
        self.landmarks_dir = getattr(config, 'landmarks_dir', '') or None
        if not self.landmarks_dir:
            raise SystemExit('ours-without-roll 管线要求 landmarks_dir（landmarks 模式）')
        self.face_model_root = Path(config.face_model_root)
        self.pnp_points = int(getattr(config, 'pnp_points', 6))
        if self.pnp_points not in (6, 28):
            raise SystemExit(f'pnp_points 只支持 6 / 28，配置为 {self.pnp_points}')
        gen = np.loadtxt(Path(__file__).parents[1] / 'zhang2015-insightface'
                         / 'face_model_xgaze.txt')
        self.gen6 = gen[[20, 23, 26, 29, 15, 19], :]
        self.calibs = {c: self._load_calib(c) for c in range(self.NUM_CAMS)}
        self._model_cache = {}
        self.n_specific = self.n_fallback = 0

    def _load_calib(self, cam):
        fs = cv2.FileStorage(str(Path(self.config.calib_dir) / f'cam{cam:02d}.xml'),
                             cv2.FILE_STORAGE_READ)
        K = fs.getNode('Camera_Matrix').mat()
        dist = fs.getNode('Distortion_Coefficients').mat()
        fs.release()
        return K, dist

    def _geometry_for(self, subject_index, cam):
        """(PnP 模型, 归一化模型, 特征点索引)；个性化缺失回退通用（与 v2 协议一致）"""
        key = (subject_index, cam)
        if key not in self._model_cache:
            sub = self.face_model_root / f'subject{subject_index:04d}'
            p6 = sub / f'cam{cam:02d}_model6.txt'
            if p6.is_file():
                m6 = np.loadtxt(p6).astype(float)
                if self.pnp_points == 28 and (sub / f'cam{cam:02d}_model28.txt').is_file():
                    self._model_cache[key] = (np.loadtxt(
                        sub / f'cam{cam:02d}_model28.txt').astype(float), m6, RIGID)
                else:
                    self._model_cache[key] = (m6, m6, self.IDX6)
                self.n_specific += 1
            else:
                self._model_cache[key] = (self.gen6, self.gen6, self.IDX6)
                self.n_fallback += 1
        return self._model_cache[key]

    def load_annotations(self, subject_index):
        gt = {}
        with open(Path(self.config.annotation_dir) / f'subject{subject_index:04d}.csv') as f:
            for line in f:
                p = line.strip().split(',')
                if len(p) < 7 or not p[0].startswith('frame'):
                    continue
                gt[(int(p[0][5:]), int(p[1][3:5]))] = \
                    [float(p[4]), float(p[5]), float(p[6])]
        return gt

    def process_subject(self, subject_index, recorder: FailureRecorder):
        subject_dir = Path(self.config.raw_data_dir) / self.config.sub_folder \
            / f'subject{subject_index:04d}'
        gaze_pts = self.load_annotations(subject_index)

        with h5py.File(Path(self.landmarks_dir) / f'subject{subject_index:04d}.h5', 'r') as f:
            fr = f['frame_index'][:].ravel()
            ci = f['cam_index'][:].ravel()
            lm_cache = f['facial_landmarks_2d'][:]
        by_frame = {}
        for r in range(len(fr)):
            by_frame.setdefault(int(fr[r]), []).append((int(ci[r]), r))
        frame_dirs = [subject_dir / f'frame{fi:04d}' for fi in sorted(by_frame)]
        if getattr(self.config, 'max_frames', 0):
            frame_dirs = frame_dirs[:self.config.max_frames]

        out_path = Path(self.config.output_dir) / f'subject{subject_index:04d}.h5'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        h5 = h5py.File(out_path, 'w')
        n_max = len(fr)
        dsets = self._create_datasets(h5, n_max)
        n = 0
        t_sub = time.time()

        q = queue.Queue(maxsize=2)

        def read_frames():
            try:
                with ThreadPoolExecutor(max_workers=self.config.num_read_workers) as ex:
                    for fd in frame_dirs:
                        cams = [c for c, _ in by_frame[int(fd.name[5:])]]
                        imgs = dict(zip(cams, ex.map(
                            lambda c: cv2.imread(str(fd / f'cam{c:02d}.JPG')), cams)))
                        q.put((fd, imgs))
                q.put(None)
            except Exception as e:
                q.put(e)

        threading.Thread(target=read_frames, daemon=True).start()

        pbar = tqdm(total=len(frame_dirs), desc=f'subject{subject_index:04d}',
                    unit='帧', ncols=100, leave=False)
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                fd, imgs = item
                frame_index = int(fd.name[5:])
                for cam, row in by_frame[frame_index]:
                    img = imgs.get(cam)
                    if img is None:
                        recorder.add(subject_index,
                                     f'frame{frame_index:04d}/cam{cam:02d}',
                                     'imread_failed')
                        continue
                    if cam in self.FLIP_CAMERAS:
                        img = cv2.rotate(img, cv2.ROTATE_180)
                    lm106 = lm_cache[row]
                    gaze_point = gaze_pts.get((frame_index, cam))
                    if gaze_point is None:
                        recorder.add(subject_index,
                                     f'frame{frame_index:04d}/cam{cam:02d}',
                                     'no_annotation')
                        continue
                    K, dist = self.calibs[cam]
                    model_pnp, model_norm, lm_idx = self._geometry_for(subject_index, cam)
                    landmarks2d = lm106[lm_idx, :].reshape(len(lm_idx), 1, 2).astype(float)
                    rvec, tvec = estimateHeadPose(landmarks2d, model_pnp, K, dist)
                    img_warped, hr_norm, gc_normalized = normalizeData_face(
                        img, model_norm, rvec, tvec, gaze_point, K,
                        fixed_forward=True)[:3]
                    R = cv2.Rodrigues(hr_norm)[0] @ cv2.Rodrigues(rvec)[0].T
                    g_theta, g_phi = vector_to_angles(gc_normalized.flatten())
                    gc_head = cv2.Rodrigues(hr_norm)[0].T @ gc_normalized
                    h_theta, h_phi = vector_to_angles(gc_head.flatten())

                    dsets['frame_index'][n] = frame_index
                    dsets['cam_index'][n] = cam
                    dsets['face_patch'][n] = img_warped
                    dsets['face_mat_norm'][n] = R
                    dsets['facial_landmarks_2d'][n] = lm106
                    dsets['face_gaze'][n] = (g_theta, g_phi)
                    dsets['face_gaze_head'][n] = (h_theta, h_phi)
                    n += 1
                pbar.update(1)
        finally:
            pbar.close()
            for d in dsets.values():
                d.resize((n,) + d.shape[1:])
            h5.close()

        log.info(f'subject{subject_index:04d} 完成: {n} 样本 -> {out_path.name} '
                 f'({(time.time() - t_sub) / 60:.1f} min)')
        return n

    @staticmethod
    def _create_datasets(h5, n_max):
        return {
            'frame_index': h5.create_dataset('frame_index', (n_max, 1), np.int32,
                                             chunks=(1, 1), maxshape=(None, 1)),
            'cam_index': h5.create_dataset('cam_index', (n_max, 1), np.int32,
                                           chunks=(1, 1), maxshape=(None, 1)),
            'face_patch': h5.create_dataset('face_patch', (n_max, 224, 224, 3),
                                            np.uint8, chunks=(1, 224, 224, 3),
                                            maxshape=(None, 224, 224, 3),
                                            compression='lzf'),
            'face_mat_norm': h5.create_dataset('face_mat_norm', (n_max, 3, 3),
                                               float, chunks=(1, 3, 3),
                                               maxshape=(None, 3, 3)),
            'facial_landmarks_2d': h5.create_dataset(
                'facial_landmarks_2d', (n_max, 106, 2), np.float32,
                chunks=(1, 106, 2), maxshape=(None, 106, 2)),
            'face_gaze': h5.create_dataset('face_gaze', (n_max, 2), float,
                                           chunks=(1, 2), maxshape=(None, 2)),
            'face_gaze_head': h5.create_dataset('face_gaze_head', (n_max, 2), float,
                                                chunks=(1, 2), maxshape=(None, 2)),
        }

    def run(self, recorder: FailureRecorder):
        dataset_path = Path(self.landmarks_dir)
        subjects = self.config.subjects if self.config.subjects else sorted(
            int(d.name[7:-3]) for d in dataset_path.glob('subject*.h5'))
        log.info(f'ours-without-roll: {len(subjects)} 受试者, 输出 {self.config.output_dir}')
        total = 0
        t_all = time.time()
        overall = tqdm(subjects, desc='xgaze 全集', unit='人', ncols=100, leave=True)
        for subject_index in overall:
            try:
                total += self.process_subject(subject_index, recorder)
            except Exception as e:
                log.error(f'[错误] subject{subject_index:04d}: {type(e).__name__}: {e}')
                recorder.add(subject_index, '<subject>',
                             f'error:{type(e).__name__}:{e}')
            overall.set_postfix({'累计样本': total})
        overall.close()
        log.info(f'全部完成: {len(subjects)} 人, {total} 样本, '
                 f'个性化 {self.n_specific} / 回退 {self.n_fallback} 相机组, '
                 f'总耗时 {(time.time() - t_all) / 3600:.2f} 小时')
        return total


def run(config, recorder):
    """preprocess.py 入口适配"""
    return OursWithRollXGazePreprocessor(config).run(recorder)
