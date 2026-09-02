"""v2 EVE 归一化：逐人 true6_canonical + DLT 头姿 + 组门控（2026-08-31）

协议：preprocess/zhang2015-specific-face-model/normalization_protocol.md
优化：按 step 分组处理，每个 mp4 只打开一次（此前每帧每相机都 open→seek→
read→close，~20000 次/人 → ~200 次/人，~100x 减少 I/O 开销）。

用法: python preprocess.py --dataset eve --method zhang2015-specific-face-model
"""
import json
import sys
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

log = get_logger('preprocess.specific_face_model.eve')

IDX6 = [35, 39, 89, 93, 78, 84]
POG_SPREAD_MAX = 5.0


def run(config, recorder):
    raw = Path(config.raw_data_dir)
    out_dir = Path(config.output_dir)
    lm_root = Path(config.landmarks_dir)
    fm_dir = Path(config.face_model_root)
    splits = config.splits if isinstance(config.splits, dict) \
        else vars(config.splits)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f'v2 EVE: true6_canonical + DLT + gate(4cam+valid+≤{POG_SPREAD_MAX:g}px)')

    for split, subjects in splits.items():
        sp_out = out_dir / split
        sp_out.mkdir(parents=True, exist_ok=True)
        sel = getattr(config, 'subjects', None) or subjects
        for subj in tqdm(sel, desc=f'{split}', unit='subj'):
            _process_subject(config, recorder, raw, lm_root / split,
                             fm_dir, sp_out, split, subj)


def _process_subject(config, recorder, raw, lm_dir, fm_dir, out_dir, split, subj):
    mpath = fm_dir / subj / 'true6_canonical.txt'
    if not mpath.is_file():
        log.warning(f'{subj}: model missing')
        return
    model = np.loadtxt(mpath)
    lm_path = lm_dir / f'{subj}.h5'
    if not lm_path.is_file():
        log.warning(f'{subj}: landmarks missing')
        return
    with h5py.File(lm_path, 'r') as f:
        fr = f['frame_index'][:].ravel()
        ci = f['cam_index'][:].ravel()
        st = f['step_index'][:].ravel()
        lm_all = f['facial_landmarks_2d'][:]
        cameras = json.loads(f.attrs['cameras'])
        steps = json.loads(f.attrs['steps'])

    Ks, Rs, ts = {}, {}, {}
    for c, cam_name in enumerate(cameras):
        for step in steps:
            p = raw / subj / step / f'{cam_name}.h5'
            if p.is_file():
                with h5py.File(p, 'r') as f:
                    Ks[c] = np.array(f['camera_matrix'], dtype=float)
                    T = np.array(f['camera_transformation'], dtype=float)
                Rs[c] = T[:3, :3]
                ts[c] = T[:3, 3].reshape(3, 1)
                break
    if len(Ks) < 3:
        log.warning(f'{subj}: calib insufficient {len(Ks)}/4')
        return

    # 帧同步
    sync_map = {}
    for r in range(len(fr)):
        c, raw_f = int(ci[r]), int(fr[r])
        sync_map.setdefault((raw_f // 2 if c == 0 else raw_f, int(st[r])),
                            []).append((c, r, raw_f))

    # ---- 预过滤：过门控的组按 step 分组 ----
    stats = {'total': 0, 'drop_cams': 0, 'drop_valid': 0, 'drop_spread': 0, 'kept': 0}
    anno_cache = {}       # (step, cam) → (validity, pog_data, mmpp)
    def _anno(step, c):
        key = (step, c)
        if key not in anno_cache:
            p = raw / subj / step / f'{cameras[c]}.h5'
            if not p.is_file():
                anno_cache[key] = None
            else:
                with h5py.File(p, 'r') as f:
                    anno_cache[key] = (
                        np.asarray(f['face_PoG_tobii/validity']),
                        np.asarray(f['face_PoG_tobii/data']),
                        np.array(f['millimeters_per_pixel'], dtype=float))
        return anno_cache[key]

    by_step = {}          # step_index → list of (sync_f, rows, pog_px)
    max_g = getattr(config, 'max_frames', 0) or 0
    for (sync_f, si), rows in sorted(sync_map.items()):
        if max_g and stats['kept'] >= max_g:
            break
        stats['total'] += 1
        step = steps[si]
        if set(c for c, _, _ in rows) != set(range(4)):
            stats['drop_cams'] += 1
            continue
        pog_px, all_ok = {}, True
        for c, r, raw_f in rows:
            a = _anno(step, c)
            if a is None or raw_f >= len(a[0]) or not a[0][raw_f]:
                all_ok = False
                break
            pog_px[c] = a[1][raw_f]
        if not all_ok:
            stats['drop_valid'] += 1
            continue
        P = np.stack([pog_px[c] for c in range(4)])
        if np.max(np.linalg.norm(P - P.mean(0), axis=1)) > POG_SPREAD_MAX:
            stats['drop_spread'] += 1
            continue
        stats['kept'] += 1
        by_step.setdefault(si, []).append((sync_f, rows, pog_px))

    # ---- h5 输出 ----
    n_est = stats['kept'] * 4
    out_path = out_dir / f'{subj}.h5'
    h5 = h5py.File(out_path, 'w')
    dsets = {
        'frame_index': h5.create_dataset('frame_index', (n_est,), np.int32,
                                         maxshape=(None,)),
        'cam_index': h5.create_dataset('cam_index', (n_est,), np.int32,
                                       maxshape=(None,)),
        'step_index': h5.create_dataset('step_index', (n_est,), np.int32,
                                        maxshape=(None,)),
        'face_patch': h5.create_dataset('face_patch', (n_est, 224, 224, 3),
                                        np.uint8, chunks=(1, 224, 224, 3),
                                        maxshape=(None, 224, 224, 3),
                                        compression='lzf'),
        'face_mat_norm': h5.create_dataset('face_mat_norm', (n_est, 3, 3), float,
                                           chunks=(1, 3, 3), maxshape=(None, 3, 3)),
        'facial_landmarks_2d': h5.create_dataset('facial_landmarks_2d',
                                                 (n_est, 106, 2), np.float32,
                                                 chunks=(1, 106, 2),
                                                 maxshape=(None, 106, 2)),
        'face_gaze': h5.create_dataset('face_gaze', (n_est, 2), float,
                                       chunks=(1, 2), maxshape=(None, 2)),
        'face_gaze_hcs': h5.create_dataset('face_gaze_hcs', (n_est, 2), float,
                                          chunks=(1, 2), maxshape=(None, 2)),
        'face_head_pose': h5.create_dataset('face_head_pose', (n_est, 2), float,
                                            chunks=(1, 2), maxshape=(None, 2)),
        'face_landmarks_3d': h5.create_dataset('face_landmarks_3d',
                                               (n_est, 6, 3), float,
                                               chunks=(1, 6, 3),
                                               maxshape=(None, 6, 3)),
    }
    written = 0

    # ---- 按 step 处理：每个 mp4 只打开一次 ----
    pbar = tqdm(by_step.items(), desc=subj[-4:], unit='step',
                leave=False, mininterval=5)
    for si, groups in pbar:
        step = steps[si]
        # 打开该 step 的 4 个 mp4（每个只开一次）+ 顺序读位置跟踪
        caps, cur_pos = {}, {}
        for c in range(4):
            mp4 = raw / subj / step / f'{cameras[c]}.mp4'
            if mp4.is_file():
                caps[c] = cv2.VideoCapture(str(mp4))
                cur_pos[c] = 0

        def _read_at(c, target):
            """顺序读帧到 target（cur_pos = 下一个待读帧的位置，从 0 起）"""
            cap = caps[c]
            if target < cur_pos[c]:            # 回退：必须 seek
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ok, img = cap.read()
                cur_pos[c] = target + 1
                return img if ok else None
            while cur_pos[c] < target:         # 前进：丢弃中间帧
                ok, _ = cap.read()
                if not ok:
                    return None
                cur_pos[c] += 1
            ok, img = cap.read()               # 读 target 帧
            cur_pos[c] += 1
            return img if ok else None

        try:
            # 帧按原始帧号排序（减少 seek 距离）
            groups.sort(key=lambda g: g[0])
            for sync_f, rows, pog_px in groups:
                # DLT + Kabsch
                rays, pv = [], []
                for c, r, raw_f in rows:
                    if c not in Ks:
                        continue
                    lm_n = cv2.undistortPoints(
                        lm_all[r][core.IDX6].astype(np.float64).reshape(-1, 1, 2),
                        Ks[c], None).reshape(-1, 2)
                    rays.append(lm_n)
                    pv.append(np.concatenate([cv2.Rodrigues(Rs[c])[0].ravel(),
                                              ts[c].ravel()]))
                if len(rays) < 3:
                    continue
                X = core.triangulate(np.stack(rays), np.stack(pv), n_points=6)
                R_head, t_head = core.kabsch(model, X)

                for c, r, raw_f in rows:
                    if c not in Ks or c not in pog_px or c not in caps:
                        continue
                    img = _read_at(c, raw_f)
                    if img is None:
                        recorder.add(f'{subj}/{step}/{cameras[c]}',
                                     f'frame{raw_f:05d}', 'decode_failed')
                        continue
                    # PoG → 相机系
                    a = _anno(step, c)
                    mmpp = a[2]
                    T4 = np.eye(4)
                    T4[:3, :3] = Rs[c]
                    T4[:3, 3:] = ts[c]
                    gp = (T4 @ np.array([pog_px[c][0] * mmpp[0],
                                         pog_px[c][1] * mmpp[1], 0., 1.])
                          )[:3].reshape(3, 1)
                    rvec = cv2.Rodrigues(Rs[c] @ R_head)[0]
                    tvec = Rs[c] @ t_head.reshape(3, 1) + ts[c]
                    img_w, hr_norm, gc_norm = normalizeData_face(
                        img, model, rvec, tvec, gp, Ks[c],
                        fixed_forward=False)[:3]
                    R = cv2.Rodrigues(hr_norm)[0] @ cv2.Rodrigues(rvec)[0].T
                    theta, phi = vector_to_angles(gc_norm.flatten())
                    hR_norm = cv2.Rodrigues(hr_norm)[0]
                    gc_hcs = hR_norm.T @ gc_norm
                    gc_hcs /= np.linalg.norm(gc_hcs)
                    hcs_t, hcs_p = vector_to_angles(gc_hcs.flatten())
                    hp, hy = head_pose_angles(hR_norm, is_true6=True)
                    X_cam = (cv2.Rodrigues(rvec)[0] @ model.T
                             + tvec.reshape(3, 1)).T

                    dsets['frame_index'][written] = raw_f
                    dsets['cam_index'][written] = c
                    dsets['step_index'][written] = si
                    dsets['face_patch'][written] = img_w
                    dsets['face_mat_norm'][written] = R
                    dsets['facial_landmarks_2d'][written] = lm_all[r]
                    dsets['face_gaze'][written] = (theta, phi)
                    dsets['face_gaze_hcs'][written] = (hcs_t, hcs_p)
                    dsets['face_head_pose'][written] = (hp, hy)
                    dsets['face_landmarks_3d'][written] = X_cam
                    written += 1
        finally:
            for cap in caps.values():
                cap.release()
        pbar.set_postfix({'written': written})

    pbar.close()
    for k in dsets:
        dsets[k].resize((written,) + dsets[k].shape[1:])
    h5.attrs['face_model'] = model
    h5.attrs['face_model_type'] = 'true6_canonical'
    h5.attrs['pipeline'] = 'zhang2015-specific-face-model v2'
    h5.attrs['steps'] = json.dumps(steps)
    h5.attrs['cameras'] = json.dumps(cameras)
    h5.close()
    log.info(f'{subj}: {written} samples | gate: total{stats["total"]} '
             f'cam{stats["drop_cams"]} valid{stats["drop_valid"]} '
             f'spread{stats["drop_spread"]} kept{stats["kept"]}')
