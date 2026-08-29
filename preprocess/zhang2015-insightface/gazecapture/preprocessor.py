"""GazeCapture 预处理（zhang2015-insightface 管线）

数据流（与 normalize_xgaze.py / normalize_mpiifacegaze.py 同构）：

    每 session（五位数目录，按官方 session 级划分归属 train/test）:
        frames.json 列出的 frames/%05d.jpg 原图
        ↓ insightface 检测 106 关键点（不使用官方 bbox/IsValid，帧有效性由
          管线成败决定：检不到脸 = no_face_detected）
        ↓ 逐帧按设备+帧分辨率加载内参（calibration/<slug>_<w>x<h>.xml，
          由 generate_calibration.py 生成；竖屏 480x640 / 横屏 640x480 逐帧二选一）
        ↓ PnP 头姿 → normalizeData_face 归一化 → 224x224 face_patch
        ↓ 视线标签: dotInfo (XCam,YCam) → CCS（dot_transfer.md 定稿公式，单位 mm）
          → 相机系 gaze_point → 归一化 → (theta, phi)
        → output_dir/<split>/<session>.h5

坐标系与单位（关键约定，详见 dot_transfer.md / obtain_camera_intrinsics.md）：
    CCS：原点前摄、+x 物理左（portrait 从右指向左）、+y 摄像头→home、单位换算成 mm
    dot → CCS（Orientation 编码 1~4）：
        Ori 1: (-XCam, -YCam)*10    Ori 2: (+XCam, +YCam)*10
        Ori 3: (-YCam, +XCam)*10    Ori 4: (+YCam, -XCam)*10     [cm→mm]
    CCS → 相机系（与 PnP 同系；前置 AVFoundation 输出不镜像，仅轴重排）：
        竖屏帧（Ori 1/2，app 旋转后，图像 x=世界右、y=世界下=物理下）:
            gaze_cam = (-ccs_x, +ccs_y, 0)        [世界轴系直接确定]
        横屏帧（Ori 3/4，传感器原生直出，图像轴=传感器读出轴）:
            gaze_cam = LANDSCAPE_SIGN * (ccs_y, ccs_x, 0)
            两种候选相差 180°（视线朝屏/背屏），LANDSCAPE_SIGN 由输出物理
            合理性定夺（视线须朝屏幕侧，见 sanity check）；默认 +1
    gaze 单位 mm 与 face model/distance_norm=600 同尺度，直接送 normalizeData_face。
"""
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml
from tqdm import tqdm

from preprocess.common import FailureRecorder
from utils.logger import get_logger
from utils.normalization import estimateHeadPose, normalizeData_face, vector_to_angles

log = get_logger('preprocess.gazecapture')

# ---- 管线常量（zhang2015-insightface，与 XGazePreprocessor 一致）----
LANDMARK_USE = [20, 23, 26, 29, 15, 19]   # 3D face model 中双眼四角+两鼻角
IDX6 = [35, 39, 89, 93, 78, 84]           # insightface 106 点对应索引
FACE_MODEL_USE = None                     # 延迟到模块初始化（避免 import 期读盘）

# 横屏帧 CCS→相机系的符号（±1）。已由 sanity check 定夺并验证（2026-08-24，
# session 00002 全量：Ori1 theta=-13.6° 俯视吻合物理；Ori3/4 phi 同号
# ——dot 恒在物理 home 侧，x_img=物理 D 时视线偏向恒定，实测 -8.8°/-15.4°
# 同号一致；若取 -1 视线将背离 dot 所在侧，荒谬）
LANDSCAPE_SIGN = 1


def _dot_to_ccs_mm(ori, xcam, ycam):
    """官方 XCam/YCam (cm) → CCS (mm)，dot_transfer.md 定稿公式"""
    if ori == 1:
        return -xcam * 10, -ycam * 10
    if ori == 2:
        return xcam * 10, ycam * 10
    if ori == 3:
        return -ycam * 10, xcam * 10
    if ori == 4:
        return ycam * 10, -xcam * 10
    raise ValueError(f'未知 Orientation: {ori}')


def _gaze_point_cam(ori, ccs_x, ccs_y):
    """CCS (mm) → 相机系 gaze 点 (mm)，与该帧 PnP 同一坐标系（见模块 docstring）"""
    if ori in (1, 2):                       # 竖屏帧
        # CCS +x（设备物理左）= 相机 +x（图像右 = 人物左），非镜像前摄
        # 2026-08-28 修正：原为 (-ccs_x, ...) 方向反了（导致全量 yaw 取反）
        p = (ccs_x, ccs_y, 0.0)
    else:                                   # 横屏帧
        s = LANDSCAPE_SIGN
        # 同理修正 x 方向：CCS +y（沿设备长边）→ 相机 x 需翻转
        p = (-s * ccs_y, s * ccs_x, 0.0)
    if ori in (2, 4):
        # ori2/ori4 的存储帧（及其上的特征点/PnP）相对 ori1/ori3 旋转了 180°，
        # 位姿的 roll 在归一化中自自洽，但注视点必须转到同一旋转后的相机轴系
        # （2026-08-27 修复：此前漏掉，导致这两类帧 pitch/yaw 双取反，~36% 标签错号）
        p = (-p[0], -p[1], 0.0)
    return p


def _slugify(device_name):
    return device_name.lower().replace(' ', '-')


class GazeCapturePreprocessor:
    """GazeCapture 数据预处理器（Zhang2015 归一化 + insightface 关键点）"""

    def __init__(self, config):
        global FACE_MODEL_USE
        self.config = config
        # landmarks 模式：从已抽取特征点 h5 索引遍历（跳过 insightface 检测）
        self.landmarks_dir = getattr(config, 'landmarks_dir', '') or None
        if FACE_MODEL_USE is None:
            fm = np.loadtxt(Path(__file__).parent.parent / 'face_model_xgaze.txt')
            FACE_MODEL_USE = fm.reshape(50, 1, 3)[LANDMARK_USE, :]
        self.face_model_use = FACE_MODEL_USE

        # 官方 session 级划分
        split_cfg = yaml.safe_load(open(config.split_file))
        self.split_sessions = {s: [f'{int(x):05d}' for x in ids]
                               for s, ids in split_cfg.items() if s != 'excluded'}
        self.excluded = split_cfg.get('excluded', [])

        # 设备→组（front_cameras.yaml），用于诊断日志；内参直接按设备加载
        fc = yaml.safe_load(open(Path(__file__).parent / 'front_cameras.yaml'))
        self.device_group = {d: v['group'] for d, v in fc['devices'].items()}

        # 内参缓存：<slug>_<w>x<h> → (K, dist)
        self._calib = {}

        self.splits = list(config.splits)
        self.session_filter = set(config.sessions) if config.sessions else None

    # ------------------------------------------------------------- 内参
    def _model_for(self, session, ori):
        """逐 (session, 朝向) 的 PnP/归一化模型。基类：通用模型（zhang2015-insightface
        语义）；zhang2015-specific-face-model 管线覆写注入 ori{o}_model6（缺失回退通用）。"""
        return self.face_model_use

    def _load_calib(self, device, w, h):
        key = (device, w, h)
        if key not in self._calib:
            path = Path(self.config.calibration_dir) / f'{_slugify(device)}_{w}x{h}.xml'
            fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
            K = fs.getNode('Camera_Matrix').mat()
            dist = fs.getNode('Distortion_Coefficients').mat()
            fs.release()
            self._calib[key] = (K, dist)
        return self._calib[key]

    # ------------------------------------------------------------ 单 session
    def process_session(self, session, split, app, recorder: FailureRecorder):
        rec_dir = Path(self.config.raw_data_dir) / session
        try:
            frames_list = json.load(open(rec_dir / 'frames.json'))
            dot = json.load(open(rec_dir / 'dotInfo.json'))
            device = json.load(open(rec_dir / 'info.json'))['DeviceName']
        except Exception as e:
            recorder.add(session, '<session>', f'error:{type(e).__name__}:{e}')
            return 0
        # frame_index -> dotInfo/screen 数组位置（frames.json 顺序与 json 数组对齐）
        pos_of = {int(name.split('.')[0]): i for i, name in enumerate(frames_list)}

        lm_cache, ori_arr = None, None
        if self.landmarks_dir:
            with h5py.File(Path(self.landmarks_dir) / split /
                           f'{session}.h5', 'r') as f:
                fi = f['frame_index'][:].ravel()
                ori_arr = f['orientation'][:].ravel()
                lm_cache = f['facial_landmarks_2d'][:]
            work = [(r, int(fi[r])) for r in range(len(fi))]   # (行号, frame_index)
        else:
            screen = json.load(open(rec_dir / 'screen.json'))
            n = min(len(frames_list), len(dot['XCam']), len(screen['Orientation']))
            if getattr(self.config, 'max_frames', 0):
                n = min(n, self.config.max_frames)      # 调试：只处理前 N 帧
            work = list(range(n))                       # 数组位置
        n = len(work)                                   # 两种模式的统一规模

        out_path = Path(self.config.output_dir) / split / f'{session}.h5'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        h5 = h5py.File(out_path, 'w')
        dsets = {
            'frame_index': h5.create_dataset('frame_index', (n, 1), np.int32,
                                             chunks=(1, 1), maxshape=(None, 1)),
            'orientation': h5.create_dataset('orientation', (n, 1), np.int8,
                                             chunks=(1, 1), maxshape=(None, 1)),
            'face_patch': h5.create_dataset('face_patch', (n, 224, 224, 3), np.uint8,
                                            chunks=(1, 224, 224, 3),
                                            maxshape=(None, 224, 224, 3),
                                            compression='lzf'),
            'face_mat_norm': h5.create_dataset('face_mat_norm', (n, 3, 3), float,
                                               chunks=(1, 3, 3), maxshape=(None, 3, 3)),
            'facial_landmarks_2d': h5.create_dataset(
                'facial_landmarks_2d', (n, 106, 2), np.float32,
                chunks=(1, 106, 2), maxshape=(None, 106, 2)),
            'face_gaze': h5.create_dataset('face_gaze', (n, 2), float,
                                           chunks=(1, 2), maxshape=(None, 2)),
        }

        written = 0
        t_sub = time.time()

        # 读图-处理流水线（与 xgaze 版同构）：8 线程预读，主线程检测/PnP/归一化
        q = queue.Queue(maxsize=2)

        def read_frames():
            try:
                with ThreadPoolExecutor(
                        max_workers=self.config.num_read_workers) as ex:
                    for it in work:
                        name = (frames_list[it] if isinstance(it, int)
                                else f'{it[1]:05d}.jpg')
                        img = cv2.imread(str(rec_dir / 'frames' / name))
                        q.put((it, img))
                q.put(None)
            except Exception as e:
                q.put(e)

        threading.Thread(target=read_frames, daemon=True).start()

        pbar = tqdm(total=len(work), desc=f'{session}({split})', unit='帧',
                    ncols=100, leave=False)
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                it, img = item
                if isinstance(it, int):              # 原始模式：数组位置
                    frame_index, frame_name = None, frames_list[it]
                    lm106 = None
                    if img is None:
                        recorder.add(session, frame_name, 'imread_failed')
                        pbar.update(1)
                        continue
                    faces = app.get(img)
                    if len(faces) == 0:
                        recorder.add(session, frame_name, 'no_face_detected')
                        pbar.update(1)
                        continue
                    lm106 = faces[0].landmark_2d_106
                    ori = screen['Orientation'][it]
                    pos = it
                else:                                # landmarks 模式：(行号, frame_index)
                    row, frame_index = it
                    frame_name = f'{frame_index:05d}.jpg'
                    if img is None:
                        recorder.add(session, frame_name, 'imread_failed')
                        pbar.update(1)
                        continue
                    lm106 = lm_cache[row]
                    ori = int(ori_arr[row])
                    pos = pos_of[frame_index]
                xcam, ycam = dot['XCam'][pos], dot['YCam'][pos]
                if dot['DotNum'][pos] == -1 or xcam is None or ycam is None:
                    recorder.add(session, frame_name, 'no_annotation')
                    pbar.update(1)
                    continue
                ccs_x, ccs_y = _dot_to_ccs_mm(ori, xcam, ycam)
                if ccs_y <= 0:
                    recorder.add(session, frame_name, 'invalid_dot')  # 朝向过渡帧噪声
                    pbar.update(1)
                    continue

                h, w = img.shape[:2]
                K, dist = self._load_calib(device, w, h)
                model = self._model_for(session, ori)
                pts2d = lm106[IDX6].reshape(6, 1, 2).astype(float)
                rvec, tvec = estimateHeadPose(pts2d, model, K, dist)
                gaze_point = _gaze_point_cam(ori, ccs_x, ccs_y)
                img_warped, hr_norm, gc_normalized = normalizeData_face(
                    img, model, rvec, tvec, np.array(gaze_point),
                    K)[:3]
                R = cv2.Rodrigues(hr_norm)[0] @ cv2.Rodrigues(rvec)[0].T
                g_theta, g_phi = vector_to_angles(gc_normalized.flatten())

                dsets['frame_index'][written] = int(frame_name.split('.')[0])
                dsets['orientation'][written] = ori
                dsets['face_patch'][written] = img_warped
                dsets['face_mat_norm'][written] = R
                dsets['facial_landmarks_2d'][written] = lm106
                dsets['face_gaze'][written] = (g_theta, g_phi)
                written += 1
                pbar.update(1)
                pbar.set_postfix({'入库': written})
        finally:
            pbar.close()
            for d in dsets.values():
                d.resize((written,) + d.shape[1:])
            h5.close()

        log.info(f'{session}({split}, {device}/{self.device_group.get(device, "?")}): '
                 f'{written}/{len(work)} 候选入库 -> {out_path.name} '
                 f'({(time.time() - t_sub) / 60:.1f} min)')
        return written

    # ------------------------------------------------------------------ 入口
    def run(self, recorder: FailureRecorder):
        app = None
        if self.landmarks_dir:
            log.info(f"landmarks 模式: 索引遍历 {self.landmarks_dir}（跳过 insightface 检测）")
        else:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis('buffalo_l',
                               allowed_modules=['detection', 'landmark_2d_106'],
                               providers=self.config.providers)
            app.prepare(ctx_id=0)

        total_n = 0
        t_all = time.time()
        for split in self.splits:
            sessions = self.split_sessions[split]
            if self.landmarks_dir:   # 只保留有 landmarks 清单的 session
                sessions = [s for s in sessions
                            if (Path(self.landmarks_dir) / split /
                                f'{s}.h5').is_file()]
            if self.session_filter:
                sessions = [s for s in sessions if s in self.session_filter]
            log.info(f'===== split {split}: {len(sessions)} 个 session，'
                     f'输出 {Path(self.config.output_dir) / split} =====')
            overall = tqdm(sessions, desc=f'gc-{split}', unit='session',
                           ncols=100, leave=True)
            for session in overall:
                try:
                    total_n += self.process_session(session, split, app, recorder)
                except Exception as e:  # 单 session 失败不中断整体
                    log.error(f'[错误] {session}: {type(e).__name__}: {e}')
                    recorder.add(session, '<session>',
                                 f'error:{type(e).__name__}:{e}')
                overall.set_postfix({'累计样本': total_n})
            overall.close()
        log.info(f'全部完成: 共 {total_n} 样本, '
                 f'总耗时 {(time.time() - t_all) / 3600:.2f} 小时')
        return total_n


def run(config, recorder):
    """preprocess.py 入口适配"""
    return GazeCapturePreprocessor(config).run(recorder)
