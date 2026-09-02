"""v2 XGaze 归一化：逐人 true6_canonical + DLT 头姿（2026-08-31）

简洁架构：逐被试逐帧顺序处理；每帧 18 张图用线程池并发读取（同目录磁盘
连续，仅压缩 I/O 等待时间）；批量 h5 写。

用法: python preprocess.py --dataset xgaze --method zhang2015-specific-face-model
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'get_face_model'))

import cv2
import h5py
import numpy as np
from tqdm import tqdm

import face_model_core as core
from utils.logger import get_logger
from utils.normalization import (head_pose_angles, normalizeData_face,
                                 vector_to_angles)

log = get_logger('preprocess.specific_face_model.xgaze')

IDX6 = [35, 39, 89, 93, 78, 84]
FLIP_CAMERAS = [3, 6, 13]
BATCH_SIZE = 50


def run(config, recorder):
    raw = Path(config.raw_data_dir) / config.sub_folder
    ann_dir = Path(config.annotation_dir)
    calib_dir = Path(config.calib_dir)
    out_dir = Path(config.output_dir)
    lm_dir = Path(config.landmarks_dir)
    fm_dir = Path(config.face_model_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    KS, DIST, ROT, TR = {}, {}, {}, {}
    for c in range(18):
        fs = cv2.FileStorage(str(calib_dir / f'cam{c:02d}.xml'),
                             cv2.FILE_STORAGE_READ)
        KS[c] = fs.getNode('Camera_Matrix').mat()
        DIST[c] = fs.getNode('Distortion_Coefficients').mat()
        ROT[c] = fs.getNode('cam_rotation').mat()
        TR[c] = fs.getNode('cam_translation').mat().reshape(3, 1)
        fs.release()

    subs = sorted(p.stem for p in lm_dir.glob('subject*.h5'))
    if getattr(config, 'subjects', None):
        subs = [f'subject{s:04d}' for s in config.subjects]
    max_f = getattr(config, 'max_frames', 0) or 0
    log.info(f'v2 XGaze: {len(subs)} subjects')

    pool = ThreadPoolExecutor(max_workers=18)    # 仅用于并发读图

    def _read(path, cam):
        img = cv2.imread(str(path))
        if img is not None and cam in FLIP_CAMERAS:
            img = cv2.rotate(img, cv2.ROTATE_180)
        return img

    for sid in tqdm(subs, desc='xgaze', unit='subj'):
        mpath = fm_dir / sid / 'true6_canonical.txt'
        if not mpath.is_file():
            log.warning(f'{sid}: model missing, skip')
            continue
        model = np.loadtxt(mpath)
        ann = {}
        with open(ann_dir / f'{sid}.csv') as f:
            for line in f:
                p = line.strip().split(',')
                ann[(p[0], p[1])] = np.array(
                    [float(p[4]), float(p[5]), float(p[6])]).reshape(3, 1)
        with h5py.File(lm_dir / f'{sid}.h5', 'r') as f:
            fr = f['frame_index'][:].ravel()
            ci = f['cam_index'][:].ravel()
            lm_all = f['facial_landmarks_2d'][:]
        by_frame = {}
        for r in range(len(fr)):
            by_frame.setdefault(int(fr[r]), []).append((int(ci[r]), r))
        frames = sorted(by_frame)
        if max_f:
            frames = frames[:max_f]

        n = len(frames) * 18
        out_path = out_dir / f'{sid}.h5'
        h5 = h5py.File(out_path, 'w')
        dsets = {
            'frame_index': h5.create_dataset('frame_index', (n, 1), np.int32,
                                             chunks=(BATCH_SIZE, 1), maxshape=(None, 1)),
            'cam_index': h5.create_dataset('cam_index', (n, 1), np.int32,
                                           chunks=(BATCH_SIZE, 1), maxshape=(None, 1)),
            'face_patch': h5.create_dataset('face_patch', (n, 224, 224, 3), np.uint8,
                                            chunks=(BATCH_SIZE, 224, 224, 3),
                                            maxshape=(None, 224, 224, 3),
                                            compression='lzf'),
            'face_mat_norm': h5.create_dataset('face_mat_norm', (n, 3, 3), float,
                                               chunks=(BATCH_SIZE, 3, 3), maxshape=(None, 3, 3)),
            'facial_landmarks_2d': h5.create_dataset('facial_landmarks_2d', (n, 106, 2),
                                                     np.float32, chunks=(BATCH_SIZE, 106, 2),
                                                     maxshape=(None, 106, 2)),
            'face_gaze': h5.create_dataset('face_gaze', (n, 2), float,
                                           chunks=(BATCH_SIZE, 2), maxshape=(None, 2)),
            'face_gaze_hcs': h5.create_dataset('face_gaze_hcs', (n, 2), float,
                                              chunks=(BATCH_SIZE, 2), maxshape=(None, 2)),
            'face_head_pose': h5.create_dataset('face_head_pose', (n, 2), float,
                                                chunks=(BATCH_SIZE, 2), maxshape=(None, 2)),
            'face_landmarks_3d': h5.create_dataset('face_landmarks_3d', (n, 6, 3), float,
                                                   chunks=(BATCH_SIZE, 6, 3), maxshape=(None, 6, 3)),
        }

        written = 0
        batch = {k: [] for k in dsets}

        def _flush():
            nonlocal written
            if not batch['face_patch']:
                return
            cnt = len(batch['face_patch'])
            start = written - cnt            # 回退到本批次的起始位置
            for k in dsets:
                if batch[k]:
                    arr = np.stack(batch[k])
                    if dsets[k].ndim > 1 and arr.ndim < dsets[k].ndim:
                        arr = arr.reshape((cnt,) + dsets[k].shape[1:])
                    dsets[k][start:start + cnt] = arr
                batch[k].clear()

        for fidx in tqdm(frames, desc=sid[-4:], unit='frame',
                         leave=False, mininterval=5):
            rows = by_frame[fidx]
            # DLT（纯计算）
            rays, pv = [], []
            for c, r in rows:
                if c >= 10:
                    continue
                lm_n = cv2.undistortPoints(
                    lm_all[r][core.IDX6].astype(np.float64).reshape(-1, 1, 2),
                    KS[c], DIST[c]).reshape(-1, 2)
                rays.append(lm_n)
                pv.append(np.concatenate([cv2.Rodrigues(ROT[c])[0].ravel(),
                                          TR[c].ravel()]))
            if len(rays) < 6:
                recorder.add(sid, f'frame{fidx:04d}', 'dlt_insufficient')
                continue
            X_w = core.triangulate(np.stack(rays), np.stack(pv), n_points=6)
            R_head, t_head = core.kabsch(model, X_w)

            # 并发读 18 张图（同目录连续，仅压缩 I/O 等待）
            paths = {}
            for cam, r in rows:
                if ann.get((f'frame{fidx:04d}', f'cam{cam:02d}.JPG')) is not None:
                    paths[cam] = raw / sid / f'frame{fidx:04d}' / f'cam{cam:02d}.JPG'
            if paths:
                futs = {cam: pool.submit(_read, p, cam)
                        for cam, p in paths.items()}
                imgs = {cam: fut.result() for cam, fut in futs.items()}
            else:
                imgs = {}

            # 逐相机顺序处理
            for cam, r in rows:
                gp = ann.get((f'frame{fidx:04d}', f'cam{cam:02d}.JPG'))
                if gp is None:
                    continue
                img = imgs.get(cam)
                if img is None:
                    recorder.add(sid, f'frame{fidx:04d}/cam{cam:02d}', 'imread_failed')
                    continue
                rvec = cv2.Rodrigues(ROT[cam] @ R_head)[0]
                tvec = ROT[cam] @ t_head.reshape(3, 1) + TR[cam]
                img_w, hr_norm, gc_norm = normalizeData_face(
                    img, model, rvec, tvec, gp, KS[cam], fixed_forward=False)[:3]
                R = cv2.Rodrigues(hr_norm)[0] @ cv2.Rodrigues(rvec)[0].T
                theta, phi = vector_to_angles(gc_norm.flatten())
                hR_norm = cv2.Rodrigues(hr_norm)[0]
                gc_hcs = hR_norm.T @ gc_norm
                gc_hcs /= np.linalg.norm(gc_hcs)
                hcs_t, hcs_p = vector_to_angles(gc_hcs.flatten())
                hp, hy = head_pose_angles(hR_norm, is_true6=True)
                X_cam = (cv2.Rodrigues(rvec)[0] @ model.T + tvec.reshape(3, 1)).T

                batch['frame_index'].append(fidx)
                batch['cam_index'].append(cam)
                batch['face_patch'].append(img_w)
                batch['face_mat_norm'].append(R)
                batch['facial_landmarks_2d'].append(lm_all[r])
                batch['face_gaze'].append((theta, phi))
                batch['face_gaze_hcs'].append((hcs_t, hcs_p))
                batch['face_head_pose'].append((hp, hy))
                batch['face_landmarks_3d'].append(X_cam)
                written += 1
                if len(batch['face_patch']) >= BATCH_SIZE:
                    _flush()

        _flush()
        for k in dsets:
            dsets[k].resize((written,) + dsets[k].shape[1:])
        h5.attrs['face_model'] = model
        h5.attrs['face_model_type'] = 'true6_canonical'
        h5.attrs['pipeline'] = 'zhang2015-specific-face-model v2'
        h5.close()
        log.info(f'{sid}: {written} samples')

    pool.shutdown()
