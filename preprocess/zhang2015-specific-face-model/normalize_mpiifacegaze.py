"""v2 MPIIFaceGaze 归一化：通用 gen_xe6 + 单目 PnP（2026-08-31）

协议：preprocess/zhang2015-specific-face-model/normalization_protocol.md
与 v1 差异：模型 = gen_xe6_canonical.txt（非 gen6）。其余同 v1。

h5 输出（平台统一格式，BGR + lzf）。

用法: python preprocess.py --dataset mpiifacegaze --method zhang2015-specific-face-model
"""
import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT))

import cv2
import h5py
import numpy as np
import scipy.io as sio
from tqdm import tqdm

from preprocess.common import FailureRecorder
from utils.logger import get_logger
from utils.normalization import (estimateHeadPose, head_pose_angles,
                                 normalizeData_face, vector_to_angles)

log = get_logger('preprocess.specific_face_model.mpiifacegaze')

IDX6 = [35, 39, 89, 93, 78, 84]


def run(config, recorder: FailureRecorder):
    raw = Path(config.raw_data_dir)
    out_dir = Path(config.output_dir)
    lm_dir = Path(config.landmarks_dir)
    GEN_XE6 = np.loadtxt(
        _PROJECT / 'preprocess/zhang2015-specific-face-model/get_face_model'
                    '/gen_xe6_canonical.txt')
    out_dir.mkdir(parents=True, exist_ok=True)
    subs = sorted(p.stem for p in lm_dir.glob('*.h5'))
    if getattr(config, 'subjects', None):
        subs = [s for s in subs if s in config.subjects]
    log.info(f'v2 MPIIFaceGaze: {len(subs)} 被试, gen_xe6 + PnP')

    for subj in tqdm(subs, desc='subjects', unit='subj'):
        _process_subject(config, recorder, raw, lm_dir, out_dir, GEN_XE6, subj)


def _process_subject(config, recorder, raw, lm_dir, out_dir, GEN_XE6, subj):
    lm_path = lm_dir / f'{subj}.h5'
    if not lm_path.is_file():
        recorder.add(subj, '<subject>', 'no_landmarks')
        return
    with h5py.File(lm_path, 'r') as f:
        lm_all = f['facial_landmarks_2d'][:]
        names = [n.decode() if isinstance(n, bytes) else str(n)
                 for n in f['image_name'][:]]
        days = f['day_index'][:].ravel()
    mat = sio.loadmat(raw / subj / 'Calibration' / 'Camera.mat')
    K = np.array(mat['cameraMatrix'], dtype=float)
    dist = np.array(mat['distCoeffs'], dtype=float).ravel()
    gpt = {}
    with open(raw / subj / f'{subj}.txt') as f:
        for line in f:
            p = line.split()
            gpt[p[0]] = np.array([float(p[24]), float(p[25]),
                                  float(p[26])]).reshape(3, 1)

    out_path = out_dir / f'{subj}.h5'
    n = len(names)
    h5 = h5py.File(out_path, 'w')
    dsets = {
        'day_index': h5.create_dataset('day_index', (n,), np.int32,
                                       maxshape=(None,)),
        'image_name': h5.create_dataset('image_name', (n,),
                                        h5py.special_dtype(vlen=str),
                                        maxshape=(None,)),
        'face_patch': h5.create_dataset('face_patch', (n, 224, 224, 3), np.uint8,
                                        chunks=(1, 224, 224, 3),
                                        maxshape=(None, 224, 224, 3),
                                        compression='lzf'),
        'face_mat_norm': h5.create_dataset('face_mat_norm', (n, 3, 3), float,
                                           chunks=(1, 3, 3), maxshape=(None, 3, 3)),
        'facial_landmarks_2d': h5.create_dataset('facial_landmarks_2d', (n, 106, 2),
                                                 np.float32, chunks=(1, 106, 2),
                                                 maxshape=(None, 106, 2)),
        'face_gaze': h5.create_dataset('face_gaze', (n, 2), float,
                                       chunks=(1, 2), maxshape=(None, 2)),
        'face_gaze_hcs': h5.create_dataset('face_gaze_hcs', (n, 2), float,
                                          chunks=(1, 2), maxshape=(None, 2)),
        'face_head_pose': h5.create_dataset('face_head_pose', (n, 2), float,
                                            chunks=(1, 2), maxshape=(None, 2)),
        'face_landmarks_3d': h5.create_dataset('face_landmarks_3d', (n, 6, 3), float,
                                               chunks=(1, 6, 3), maxshape=(None, 6, 3)),
    }
    written = 0
    for i in tqdm(range(len(names)), desc=subj, unit='img',
                 leave=False, mininterval=5):
        day = int(days[i])
        fname = names[i]
        key = f'day{day:02d}/{fname}'
        gp = gpt.get(key)
        if gp is None:
            recorder.add(subj, fname, 'no_annotation')
            continue
        img = cv2.imread(str(raw / subj / f'day{day:02d}' / fname))
        if img is None:
            recorder.add(subj, fname, 'imread_failed')
            continue
        try:
            rvec, tvec = estimateHeadPose(
                lm_all[i][IDX6].reshape(6, 1, 2).astype(float),
                GEN_XE6, K, dist)
        except cv2.error:
            recorder.add(subj, fname, 'pnp_failed')
            continue
        img_w, hr_norm, gc_norm = normalizeData_face(
            img, GEN_XE6, rvec, tvec, gp, K, fixed_forward=False)[:3]
        R = cv2.Rodrigues(hr_norm)[0] @ cv2.Rodrigues(rvec)[0].T
        theta, phi = vector_to_angles(gc_norm.flatten())
        dsets['day_index'][written] = day
        dsets['image_name'][written] = fname
        dsets['face_patch'][written] = img_w
        dsets['face_mat_norm'][written] = R
        dsets['facial_landmarks_2d'][written] = lm_all[i]
        dsets['face_gaze'][written] = (theta, phi)
                # 新增字段计算与写入
        hR_norm = cv2.Rodrigues(hr_norm)[0]
        gc_hcs = hR_norm.T @ gc_norm
        gc_hcs /= np.linalg.norm(gc_hcs)
        hcs_t, hcs_p = vector_to_angles(gc_hcs.flatten())
        hp, hy = head_pose_angles(hR_norm, is_true6=True)
        X_cam = (cv2.Rodrigues(rvec)[0] @ GEN_XE6.T + tvec.reshape(3, 1)).T  # (6,3)
        dsets['face_gaze_hcs'][written] = (hcs_t, hcs_p)
        dsets['face_head_pose'][written] = (hp, hy)
        dsets['face_landmarks_3d'][written] = X_cam
        written += 1
    for k in dsets:
        dsets[k].resize((written,) + dsets[k].shape[1:])
    h5.attrs['face_model'] = GEN_XE6
    h5.attrs['face_model_type'] = 'gen_xe6'
    h5.attrs['pipeline'] = 'zhang2015-specific-face-model v2'

    h5.close()
    log.info(f'{subj}: {written} 样本 → {out_path.name}')
