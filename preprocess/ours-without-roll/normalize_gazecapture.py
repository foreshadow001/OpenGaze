"""v3 GazeCapture 归一化：ours-without-roll（fixed_forward=True，2026-09-02）

协议：preprocess/ours-without-roll/normalization_protocol.md
数据流：v2 产物（sfm/gazecapture_specific_224）逐样本恢复几何（无 PnP），
原图按实际尺寸查内参 xml 读入（同 v2），roll-only 归一化。
单目：世界系 = 相机本身（ext=None）。

用法: python preprocess.py --dataset gazecapture --method ours-without-roll
"""
import json
import sys
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

log = get_logger('preprocess.ours_without_roll.gazecapture')

BATCH_SIZE = 50


def run(config, recorder: FailureRecorder):
    v2_dir = Path(config.v2_dir)
    raw = Path(config.raw_data_dir)
    calib_dir = Path(config.calibration_dir)
    out_dir = Path(config.output_dir)
    splits = getattr(config, 'splits', None) or ['train', 'test']
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info('v3 GazeCapture: fixed_forward=True (roll-only), world = camera')

    for split in splits:
        sess = sorted(p.stem for p in (v2_dir / split).glob('*.h5'))
        if getattr(config, 'sessions', None):
            sess = [s for s in sess if s in config.sessions]
        for s in tqdm(sess, desc=split, unit='sess'):
            _session(recorder, v2_dir / split, raw, calib_dir,
                     out_dir / split, s,
                     int(getattr(config, 'max_frames', 0) or 0))


def _session(recorder, v2_dir, raw, calib_dir, out_dir, session, max_rows):
    src = v2_dir / f'{session}.h5'
    if not src.is_file():
        recorder.add(session, '<session>', 'no_v2_product')
        return
    try:
        device = json.load(open(raw / session / 'info.json'))['DeviceName'] \
            .lower().replace(' ', '-')
    except Exception as e:
        recorder.add(session, '<session>', f'error:{type(e).__name__}')
        return
    cals = {}
    for (w, h) in ((480, 640), (640, 480)):
        cal = calib_dir / f'{device}_{w}x{h}.xml'
        if cal.is_file():
            fs = cv2.FileStorage(str(cal), cv2.FILE_STORAGE_READ)
            cals[(w, h)] = fs.getNode('Camera_Matrix').mat()
            fs.release()
    if not cals:
        recorder.add(session, '<session>', 'no_calibration')
        return

    with h5py.File(src, 'r') as f:
        fr = f['frame_index'][:].ravel()
        ori = f['orientation'][:].ravel()
        gaze = f['face_gaze'][:]
        lm3d = f['face_landmarks_3d'][:]
        lm2d = f['facial_landmarks_2d'][:]
        matn = f['face_mat_norm'][:]
        model = np.array(f.attrs['face_model'])
        mtype = str(f.attrs['face_model_type'])
    rows = list(range(len(fr)))[:max_rows or None]

    out_dir.mkdir(parents=True, exist_ok=True)
    n_est = len(rows)
    out = h5py.File(out_dir / f'{session}.h5', 'w')
    dsets = {
        'frame_index': out.create_dataset('frame_index', (n_est, 1), np.int32,
                                          chunks=(BATCH_SIZE, 1), maxshape=(None, 1)),
        'orientation': out.create_dataset('orientation', (n_est, 1), np.int8,
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

    pbar = tqdm(rows, desc=session, unit='frame', leave=False, mininterval=5)
    for r in pbar:
        fidx = int(fr[r])
        fname = f'{fidx:05d}.jpg'
        img = cv2.imread(str(raw / session / 'frames' / fname))
        if img is None:
            recorder.add(session, fname, 'imread_failed')
            continue
        h, w = img.shape[:2]
        if (w, h) not in cals:
            recorder.add(session, fname, 'no_calibration')
            continue
        res = v3_core.v3_sample(img, model, lm3d[r], matn[r], gaze[r],
                                cals[(w, h)], None)
        batch['frame_index'].append(fidx)
        batch['orientation'].append(int(ori[r]))
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
    log.info(f'{session}: {written}/{n_est} 样本')
