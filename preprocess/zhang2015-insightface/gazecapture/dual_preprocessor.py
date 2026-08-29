"""GazeCapture 双输出预处理器：读图一次，同时产出 insightface 版 + specific-face-model 版

用途：全量重预处理时将两版合并为单次读图，I/O 减半（~2.2h 替代两版并行 ~4h）。
管线不变——v1 走通用模型 PnP + 归一化 → ylx；v2 走个性化模型 → sfm。
仅 sfm 名单内的 session 产出双份，名单外只产 v1。

用法（preprocess.py 入口）：
  config 增加 dual_output_dir / dual_face_model_root / dual_split_file 字段即可激活
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
from .preprocessor import (_dot_to_ccs_mm, _gaze_point_cam, _slugify,
                           IDX6, LANDMARK_USE, _create_h5_dsets)

log = get_logger('preprocess.gazecapture.dual')


class DualGazeCapturePreprocessor:
    """双输出：每帧读图一次，通用模型 → v1 h5 + 个性化模型 → v2 h5"""

    def __init__(self, config):
        self.config = config
        self.landmarks_dir = getattr(config, 'landmarks_dir', '') or None
        if not self.landmarks_dir:
            raise SystemExit('双输出模式要求 landmarks_dir（landmarks 模式）')

        # v1 通用模型
        fm = np.loadtxt(Path(__file__).parent.parent / 'face_model_xgaze.txt')
        self.gen_model = fm.reshape(50, 1, 3)[LANDMARK_USE, :]

        # v2 个性化模型根目录 + sfm 名单
        self.dual_face_model_root = Path(config.dual_face_model_root)
        dual_split = yaml.safe_load(open(config.dual_split_file))
        self.dual_sessions = set(dual_split.get('train', []) + dual_split.get('test', []))
        self.dual_output_dir = Path(config.dual_output_dir)

        # v1 session 划分
        split_cfg = yaml.safe_load(open(config.split_file))
        self.split_sessions = {s: [f'{int(x):05d}' for x in ids]
                               for s, ids in split_cfg.items() if s != 'excluded'}
        self.splits = list(config.splits)
        self.session_filter = set(config.sessions) if config.sessions else None

        # 内参缓存
        self._calib = {}
        # 个性化模型缓存 (session, ori) → model or None
        self._model_cache = {}

    def _load_calib(self, device, w, h):
        key = (device, w, h)
        if key not in self._calib:
            path = Path(self.config.calibration_dir) / f'{_slugify(device)}_{w}x{h}.xml'
            fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
            self._calib[key] = (fs.getNode('Camera_Matrix').mat(),
                                fs.getNode('Distortion_Coefficients').mat())
            fs.release()
        return self._calib[key]

    def _dual_model(self, session, ori):
        """个性化模型查找（缓存），缺失返回 None（回退通用）"""
        key = (session, int(ori))
        if key not in self._model_cache:
            p = self.dual_face_model_root / session / f'ori{int(ori)}_model6.txt'
            self._model_cache[key] = (np.loadtxt(p).astype(float)
                                      if p.is_file() else None)
        return self._model_cache[key]

    def process_session(self, session, split, recorder):
        rec_dir = Path(self.config.raw_data_dir) / session
        try:
            frames_list = json.load(open(rec_dir / 'frames.json'))
            dot = json.load(open(rec_dir / 'dotInfo.json'))
            device = json.load(open(rec_dir / 'info.json'))['DeviceName']
        except Exception as e:
            recorder.add(session, '<session>', f'error:{type(e).__name__}:{e}')
            return 0, 0
        pos_of = {int(n.split('.')[0]): i for i, n in enumerate(frames_list)}

        with h5py.File(Path(self.landmarks_dir) / split / f'{session}.h5', 'r') as f:
            fi = f['frame_index'][:].ravel()
            ori_arr = f['orientation'][:].ravel()
            lm_cache = f['facial_landmarks_2d'][:]
        work = [(r, int(fi[r])) for r in range(len(fi))]
        n = len(work)

        # 判断是否双输出
        is_dual = session in self.dual_sessions

        # v1 h5
        v1_path = Path(self.config.output_dir) / split / f'{session}.h5'
        v1_path.parent.mkdir(parents=True, exist_ok=True)
        h5_v1 = h5py.File(v1_path, 'w')
        dsets_v1 = _create_h5_dsets(h5_v1, n)

        # v2 h5（仅 sfm 名单内）
        h5_v2, dsets_v2 = None, None
        if is_dual:
            v2_path = self.dual_output_dir / split / f'{session}.h5'
            v2_path.parent.mkdir(parents=True, exist_ok=True)
            h5_v2 = h5py.File(v2_path, 'w')
            dsets_v2 = _create_h5_dsets(h5_v2, n)

        w1 = w2 = 0
        t_sub = time.time()
        q = queue.Queue(maxsize=2)

        def read_frames():
            try:
                with ThreadPoolExecutor(max_workers=self.config.num_read_workers) as ex:
                    for row, frame_idx in work:
                        name = f'{frame_idx:05d}.jpg'
                        img = cv2.imread(str(rec_dir / 'frames' / name))
                        q.put(((row, frame_idx), img))
                q.put(None)
            except Exception as e:
                q.put(e)

        threading.Thread(target=read_frames, daemon=True).start()

        pbar = tqdm(total=n, desc=f'{session}({split}{"+dual" if is_dual else ""})',
                    unit='帧', ncols=100, leave=False)
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                (row, frame_idx), img = item
                fname = f'{frame_idx:05d}.jpg'

                if img is None:
                    recorder.add(session, fname, 'imread_failed')
                    pbar.update(1)
                    continue
                lm106 = lm_cache[row]
                ori = int(ori_arr[row])
                pos = pos_of.get(frame_idx)
                if pos is None:
                    pbar.update(1)
                    continue
                xcam, ycam = dot['XCam'][pos], dot['YCam'][pos]
                if dot['DotNum'][pos] == -1 or xcam is None:
                    recorder.add(session, fname, 'no_annotation')
                    pbar.update(1)
                    continue
                ccs_x, ccs_y = _dot_to_ccs_mm(ori, xcam, ycam)
                if ccs_y <= 0:
                    recorder.add(session, fname, 'invalid_dot')
                    pbar.update(1)
                    continue

                # 共享：只读一次
                h, w = img.shape[:2]
                K, dist = self._load_calib(device, w, h)
                pts2d = lm106[IDX6].reshape(6, 1, 2).astype(float)
                gp = np.array(_gaze_point_cam(ori, ccs_x, ccs_y))

                def run_and_write(model, dsets, idx):
                    rvec, tvec = estimateHeadPose(pts2d, model, K, dist)
                    iw, hr, gc = normalizeData_face(img, model, rvec, tvec, gp, K)[:3]
                    R = cv2.Rodrigues(hr)[0] @ cv2.Rodrigues(rvec)[0].T
                    th, ph = vector_to_angles(gc.flatten())
                    dsets['frame_index'][idx] = frame_idx
                    dsets['orientation'][idx] = ori
                    dsets['face_patch'][idx] = iw
                    dsets['face_mat_norm'][idx] = R
                    dsets['facial_landmarks_2d'][idx] = lm106
                    dsets['face_gaze'][idx] = (th, ph)

                # v1：通用模型
                run_and_write(self.gen_model, dsets_v1, w1)
                w1 += 1

                # v2：个性化模型
                if dsets_v2 is not None:
                    m2 = self._dual_model(session, ori)
                    run_and_write(m2 if m2 is not None else self.gen_model,
                                  dsets_v2, w2)
                    w2 += 1

                pbar.update(1)
                pbar.set_postfix({'v1': w1, 'v2': w2})
        finally:
            pbar.close()
            for d in dsets_v1.values():
                d.resize((w1,) + d.shape[1:])
            h5_v1.close()
            if h5_v2 is not None:
                for d in dsets_v2.values():
                    d.resize((w2,) + d.shape[1:])
                h5_v2.close()

        log.info(f'{session}({split}): v1={w1} v2={w2 if is_dual else "—"}/ | '
                 f'({(time.time() - t_sub) / 60:.1f} min)')
        return w1, w2

    def run(self, recorder):
        total1 = total2 = 0
        t_all = time.time()
        for split in self.splits:
            sessions = [s for s in self.split_sessions[split]
                        if (Path(self.landmarks_dir) / split / f'{s}.h5').is_file()]
            if self.session_filter:
                sessions = [s for s in sessions if s in self.session_filter]
            log.info(f'===== {split}: {len(sessions)} session =====')
            overall = tqdm(sessions, desc=f'gc-{split}', unit='session',
                           ncols=100, leave=True)
            for session in overall:
                try:
                    n1, n2 = self.process_session(session, split, recorder)
                    total1 += n1
                    total2 += n2
                except Exception as e:
                    log.error(f'[错误] {session}: {type(e).__name__}: {e}')
                    recorder.add(session, '<session>', f'error:{type(e).__name__}:{e}')
                overall.set_postfix({'v1': total1, 'v2': total2})
            overall.close()
        log.info(f'全部完成: v1={total1} v2={total2}, '
                 f'总耗时 {(time.time() - t_all) / 3600:.2f} 小时')
        return total1
