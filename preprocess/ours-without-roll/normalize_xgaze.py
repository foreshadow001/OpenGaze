"""v3 XGaze 归一化：ours-without-roll（fixed_forward=True，2026-09-02）

协议：preprocess/ours-without-roll/normalization_protocol.md
数据流：v2 产物（sfm/xgaze_specific_224）逐样本恢复几何（Kabsch + 角度反解，
无 DLT/PnP），原图重读（FLIP_CAMERAS 180° 旋转还原物理相机，同 v2），
roll-only 归一化。字段与 v2 一致，face_head_pose → face_head_elev_azim
（世界系 = 官方穹顶系 = cam00 系）。

用法: python preprocess.py --dataset xgaze --method ours-without-roll
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from preprocess.common import FailureRecorder
from utils.logger import get_logger

import v3_core

log = get_logger('preprocess.ours_without_roll.xgaze')

FLIP_CAMERAS = [3, 6, 13]    # 存储帧倒置的相机，读图后 180° 还原（同 v2）
BATCH_SIZE = 50


def run(config, recorder: FailureRecorder):
    v2_dir = Path(config.v2_dir)
    raw = Path(config.raw_data_dir) / getattr(config, 'sub_folder', 'train')
    calib = Path(config.calib_dir)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    KS, ROT = {}, {}
    for c in range(18):
        fs = cv2.FileStorage(str(calib / f'cam{c:02d}.xml'), cv2.FILE_STORAGE_READ)
        KS[c] = fs.getNode('Camera_Matrix').mat()
        ROT[c] = fs.getNode('cam_rotation').mat()
        fs.release()

    subs = sorted(p.stem for p in v2_dir.glob('subject*.h5'))
    if getattr(config, 'subjects', None):
        subs = [f'subject{int(s):04d}' for s in config.subjects]
    max_f = int(getattr(config, 'max_frames', 0) or 0)
    pool = ThreadPoolExecutor(
        max_workers=int(getattr(config, 'num_read_workers', 18)))
    log.info(f'v3 XGaze: {len(subs)} subjects, fixed_forward=True (roll-only)')

    for sid in tqdm(subs, desc='xgaze', unit='subj'):
        _subject(recorder, v2_dir, raw, out_dir, KS, ROT, pool, sid, max_f)
    pool.shutdown(wait=True)


def _subject(recorder, v2_dir, raw, out_dir, KS, ROT, pool, sid, max_f):
    src = v2_dir / f'{sid}.h5'
    if not src.is_file():
        recorder.add(sid, '<subject>', 'no_v2_product')
        return
    with h5py.File(src, 'r') as f:
        fr = f['frame_index'][:].ravel()
        ci = f['cam_index'][:].ravel()
        gaze = f['face_gaze'][:]
        lm3d = f['face_landmarks_3d'][:]
        lm2d = f['facial_landmarks_2d'][:]
        matn = f['face_mat_norm'][:]
        model = np.array(f.attrs['face_model'])
        mtype = str(f.attrs['face_model_type'])
    by_frame = {}
    for r in range(len(fr)):
        by_frame.setdefault(int(fr[r]), []).append(r)
    frames = sorted(by_frame)
    if max_f:
        frames = frames[:max_f]
    n_est = sum(len(by_frame[fd]) for fd in frames)

    out = h5py.File(out_dir / f'{sid}.h5', 'w')
    dsets = {
        'frame_index': out.create_dataset('frame_index', (n_est, 1), np.int32,
                                          chunks=(BATCH_SIZE, 1), maxshape=(None, 1)),
        'cam_index': out.create_dataset('cam_index', (n_est, 1), np.int32,
                                        chunks=(BATCH_SIZE, 1), maxshape=(None, 1)),
        'face_patch': out.create_dataset('face_patch', (n_est, 224, 224, 3), np.uint8,
                                         chunks=(1, 224, 224, 3),   # 逐样本 chunk（批量 chunk 随机读放大 50 倍，2026-09-03）
                                         maxshape=(None, 224, 224, 3), compression='lzf'),
        'face_mat_norm': out.create_dataset('face_mat_norm', (n_est, 3, 3), float,
                                            chunks=(BATCH_SIZE, 3, 3), maxshape=(None, 3, 3)),
        'facial_landmarks_2d': out.create_dataset('facial_landmarks_2d',
                                                  (n_est, 106, 2), np.float32,
                                                  chunks=(BATCH_SIZE, 106, 2),
                                                  maxshape=(None, 106, 2)),
        'face_gaze': out.create_dataset('face_gaze', (n_est, 2), float,
                                        chunks=(BATCH_SIZE, 2), maxshape=(None, 2)),
        'face_gaze_hcs': out.create_dataset('face_gaze_hcs', (n_est, 2), float,
                                            chunks=(BATCH_SIZE, 2), maxshape=(None, 2)),
        'face_head_elev_azim': out.create_dataset('face_head_elev_azim', (n_est, 2), float,
                                                  chunks=(BATCH_SIZE, 2), maxshape=(None, 2)),
        'face_landmarks_3d': out.create_dataset('face_landmarks_3d', (n_est, 6, 3), float,
                                                chunks=(BATCH_SIZE, 6, 3), maxshape=(None, 6, 3)),
    }
    written = 0
    batch = {k: [] for k in dsets}

    def _flush():
        nonlocal written
        if not batch['face_patch']:
            return
        cnt = len(batch['face_patch'])
        start = written - cnt
        for k in dsets:
            if batch[k]:
                arr = np.stack(batch[k])
                if dsets[k].ndim > 1 and arr.ndim < dsets[k].ndim:
                    arr = arr.reshape((cnt,) + dsets[k].shape[1:])
                dsets[k][start:start + cnt] = arr
            batch[k].clear()

    def _read(path, cam):
        img = cv2.imread(str(path))
        if img is not None and cam in FLIP_CAMERAS:
            img = cv2.rotate(img, cv2.ROTATE_180)
        return img

    pbar = tqdm(frames, desc=sid[-4:], unit='frame', leave=False, mininterval=5)
    for fidx in pbar:
        rows = by_frame[fidx]
        futs = {int(ci[r]): pool.submit(
            _read, raw / sid / f'frame{fidx:04d}' / f'cam{int(ci[r]):02d}.JPG',
            int(ci[r])) for r in rows}
        imgs = {c: fu.result() for c, fu in futs.items()}
        for r in rows:
            c = int(ci[r])
            img = imgs.get(c)
            if img is None:
                recorder.add(sid, f'frame{fidx:04d}/cam{c:02d}', 'imread_failed')
                continue
            res = v3_core.v3_sample(img, model, lm3d[r], matn[r], gaze[r],
                                    KS[c], ROT[c].T)
            batch['frame_index'].append(fidx)
            batch['cam_index'].append(c)
            batch['face_patch'].append(res['patch'])
            batch['face_mat_norm'].append(res['R'])
            batch['facial_landmarks_2d'].append(lm2d[r])
            batch['face_gaze'].append(res['gaze'])
            batch['face_gaze_hcs'].append(res['hcs'])
            batch['face_head_elev_azim'].append(res['world'])
            batch['face_landmarks_3d'].append(lm3d[r])
            written += 1
            if written % BATCH_SIZE == 0:
                _flush()
        pbar.set_postfix({'written': written})
    pbar.close()
    _flush()
    for k in dsets:
        dsets[k].resize((written,) + dsets[k].shape[1:])
    out.attrs['face_model'] = model
    out.attrs['face_model_type'] = mtype
    out.attrs['pipeline'] = 'ours-without-roll v3'
    out.close()
    log.info(f'{sid}: {written}/{n_est} 样本')
