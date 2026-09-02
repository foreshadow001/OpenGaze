"""v2 GazeCapture 归一化：通用 gen_xe6 + 单目 PnP + 官方质量门（2026-08-31）

协议：preprocess/zhang2015-specific-face-model/normalization_protocol.md
与 v1 差异：
- 模型 = gen_xe6_canonical.txt（非 gen6）；
- 官方质量门：appleFace/eye IsValid 全真 + invalid_dot（同 v1 新版）。

h5 输出（平台统一格式，BGR + lzf）。

用法: python preprocess.py --dataset gazecapture --method zhang2015-specific-face-model
"""
import importlib.util
import json
import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT))

import cv2
import h5py
import numpy as np
import yaml
from tqdm import tqdm

from preprocess.common import FailureRecorder
from utils.logger import get_logger
from utils.normalization import (estimateHeadPose, head_pose_angles,
                                 normalizeData_face, vector_to_angles)

log = get_logger('preprocess.specific_face_model.gazecapture')

IDX6 = [35, 39, 89, 93, 78, 84]

# 官方 dot 链（唯一实现）
_gc_spec = importlib.util.spec_from_file_location(
    'gc_pre',
    _PROJECT / 'preprocess/zhang2015-insightface/gazecapture/preprocessor.py')
gc_pre = importlib.util.module_from_spec(_gc_spec)
_gc_spec.loader.exec_module(gc_pre)


def run(config, recorder: FailureRecorder):
    raw = Path(config.raw_data_dir)
    out_dir = Path(config.output_dir)
    calib_dir = Path(config.calibration_dir)
    lm_dir = Path(config.landmarks_dir)
    GEN_XE6 = np.loadtxt(
        _PROJECT / 'preprocess/zhang2015-specific-face-model/get_face_model'
                    '/gen_xe6_canonical.txt')
    # session 范围
    split_data = yaml.safe_load(open(config.split_file))
    splits_cfg = config.splits if hasattr(config, 'splits') else ['train', 'test']
    all_sessions = []
    for sp in splits_cfg:
        for s in split_data.get(sp, []):
            all_sessions.append((sp, s))
    if getattr(config, 'sessions', None):
        all_sessions = [(sp, s) for sp, s in all_sessions if s in config.sessions]
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f'v2 GazeCapture: {len(all_sessions)} session, gen_xe6 + PnP + 官方质量门')

    _calib = {}
    for sp, sess in tqdm(all_sessions, desc='sessions', unit='sess'):
        _process_session(config, recorder, raw, lm_dir, out_dir, calib_dir,
                         GEN_XE6, _calib, sp, sess)


def _process_session(config, recorder, raw, lm_dir, out_dir, calib_dir,
                     GEN_XE6, _calib, split, session):
    rec = raw / session
    lm_path = lm_dir / split / f'{session}.h5'
    if not lm_path.is_file():
        recorder.add(session, '<session>', 'no_landmarks')
        return
    try:
        device = json.load(open(rec / 'info.json'))['DeviceName']
        dot = json.load(open(rec / 'dotInfo.json'))
        frames_list = json.load(open(rec / 'frames.json'))
        face_v = json.load(open(rec / 'appleFace.json'))['IsValid']
        leye_v = json.load(open(rec / 'appleLeftEye.json'))['IsValid']
        reye_v = json.load(open(rec / 'appleRightEye.json'))['IsValid']
    except Exception as e:
        recorder.add(session, '<session>', f'error:{type(e).__name__}')
        return
    pos_of = {int(n.split('.')[0]): i for i, n in enumerate(frames_list)}
    slug = device.lower().replace(' ', '-')
    cals = {}
    for (w, h) in ((480, 640), (640, 480)):
        cal = calib_dir / f'{slug}_{w}x{h}.xml'
        if cal.is_file():
            fs = cv2.FileStorage(str(cal), cv2.FILE_STORAGE_READ)
            cals[(w, h)] = (fs.getNode('Camera_Matrix').mat(),
                            fs.getNode('Distortion_Coefficients').mat())
            fs.release()
    if not cals:
        recorder.add(session, '<session>', 'no_calibration')
        return

    with h5py.File(lm_path, 'r') as f:
        fr = f['frame_index'][:].ravel()
        ori_arr = f['orientation'][:].ravel()
        lm_all = f['facial_landmarks_2d'][:]

    out_path = out_dir / split / f'{session}.h5'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(fr)
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
    for r in tqdm(range(len(fr)), desc=session, unit='frame',
                 leave=False, mininterval=5):
        fidx = int(fr[r])
        ori = int(ori_arr[r])
        lm106 = lm_all[r]
        frame_name = f'{fidx:05d}.jpg'
        pi = pos_of.get(fidx)
        if pi is None or dot['DotNum'][pi] == -1:
            recorder.add(session, frame_name, 'no_annotation')
            continue
        if not (face_v[pi] and leye_v[pi] and reye_v[pi]):
            recorder.add(session, frame_name, 'apple_invalid')
            continue
        xcam, ycam = dot['XCam'][pi], dot['YCam'][pi]
        if xcam is None or ycam is None:
            recorder.add(session, frame_name, 'no_annotation')
            continue
        ccs_x, ccs_y = gc_pre._dot_to_ccs_mm(ori, xcam, ycam)
        if ccs_y <= 0:
            recorder.add(session, frame_name, 'invalid_dot')
            continue
        gp = np.array(gc_pre._gaze_point_cam(ori, ccs_x, ccs_y)).reshape(3, 1)
        img = cv2.imread(str(rec / 'frames' / frame_name))
        if img is None:
            recorder.add(session, frame_name, 'imread_failed')
            continue
        h, w = img.shape[:2]
        if (w, h) not in cals:
            recorder.add(session, frame_name, 'no_calibration')
            continue
        K, dist = cals[(w, h)]
        try:
            rvec, tvec = estimateHeadPose(
                lm106[IDX6].reshape(6, 1, 2).astype(float), GEN_XE6, K, dist)
        except cv2.error:
            recorder.add(session, frame_name, 'pnp_failed')
            continue
        img_w, hr_norm, gc_norm = normalizeData_face(
            img, GEN_XE6, rvec, tvec, gp, K, fixed_forward=False)[:3]
        R = cv2.Rodrigues(hr_norm)[0] @ cv2.Rodrigues(rvec)[0].T
        theta, phi = vector_to_angles(gc_norm.flatten())
        dsets['frame_index'][written] = fidx
        dsets['orientation'][written] = ori
        dsets['face_patch'][written] = img_w
        dsets['face_mat_norm'][written] = R
        dsets['facial_landmarks_2d'][written] = lm106
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
