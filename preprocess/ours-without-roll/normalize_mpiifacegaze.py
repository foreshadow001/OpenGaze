"""v3 MPIIFaceGaze 归一化：ours-without-roll（fixed_forward=True，2026-09-02）

协议：preprocess/ours-without-roll/normalization_protocol.md
数据流：v2 产物（sfm/mpiifacegaze_specific_224）逐样本恢复几何（无 PnP），
原图按 day_index/image_name 读入（同 v2），roll-only 归一化。
单目：世界系 = 相机本身（ext=None）。

用法: python preprocess.py --dataset mpiifacegaze --method ours-without-roll
"""
import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import h5py
import numpy as np
import scipy.io as sio
from tqdm import tqdm

from preprocess.common import FailureRecorder
from utils.logger import get_logger

import v3_core

log = get_logger('preprocess.ours_without_roll.mpiifacegaze')

BATCH_SIZE = 50


def run(config, recorder: FailureRecorder):
    v2_dir = Path(config.v2_dir)
    raw = Path(config.raw_data_dir)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    subs = sorted(p.stem for p in v2_dir.glob('*.h5'))
    if getattr(config, 'subjects', None):
        subs = [s for s in subs if s in config.subjects]
    log.info(f'v3 MPIIFaceGaze: {len(subs)} subjects, '
             'fixed_forward=True (roll-only), world = camera')

    for subj in tqdm(subs, desc='subjects', unit='subj'):
        _subject(recorder, v2_dir, raw, out_dir, subj,
                 int(getattr(config, 'max_frames', 0) or 0))


def _subject(recorder, v2_dir, raw, out_dir, subj, max_rows):
    src = v2_dir / f'{subj}.h5'
    if not src.is_file():
        recorder.add(subj, '<subject>', 'no_v2_product')
        return
    with h5py.File(src, 'r') as f:
        gaze = f['face_gaze'][:]
        lm3d = f['face_landmarks_3d'][:]
        lm2d = f['facial_landmarks_2d'][:]
        matn = f['face_mat_norm'][:]
        days = f['day_index'][:].ravel()
        names = [n.decode() if isinstance(n, bytes) else str(n)
                 for n in f['image_name'][:]]
        model = np.array(f.attrs['face_model'])
        mtype = str(f.attrs['face_model_type'])
    mat = sio.loadmat(raw / subj / 'Calibration' / 'Camera.mat')
    K = np.array(mat['cameraMatrix'], dtype=float)
    rows = list(range(len(names)))[:max_rows or None]

    n_est = len(rows)
    out = h5py.File(out_dir / f'{subj}.h5', 'w')
    dsets = {
        'day_index': out.create_dataset('day_index', (n_est,), np.int32,
                                        chunks=(BATCH_SIZE,), maxshape=(None,)),
        'image_name': out.create_dataset('image_name', (n_est,),
                                         h5py.special_dtype(vlen=str),
                                         chunks=(BATCH_SIZE,), maxshape=(None,)),
        'face_patch': out.create_dataset('face_patch', (n_est, 224, 224, 3), np.uint8,
                                         chunks=(BATCH_SIZE, 224, 224, 3),
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

    pbar = tqdm(rows, desc=subj, unit='img', leave=False, mininterval=5)
    for r in pbar:
        day, fname = int(days[r]), names[r]
        img = cv2.imread(str(raw / subj / f'day{day:02d}' / fname))
        if img is None:
            recorder.add(subj, fname, 'imread_failed')
            continue
        res = v3_core.v3_sample(img, model, lm3d[r], matn[r], gaze[r], K, None)
        batch['day_index'].append(day)
        batch['image_name'].append(fname)
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
    log.info(f'{subj}: {written}/{n_est} 样本')
