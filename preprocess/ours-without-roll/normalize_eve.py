"""v3 EVE 归一化：ours-without-roll（fixed_forward=True，2026-09-02）

协议：preprocess/ours-without-roll/normalization_protocol.md
数据流：v2 产物（sfm/eve_specific_224）逐样本恢复几何（无 DLT/PnP），原图
按 step 分组顺序读 mp4（同 v2），roll-only 归一化。世界系 = basler 相机系
（cam 0；ext = R_basler·R_c^T，同 step 各相机反解同值）。

用法: python preprocess.py --dataset eve --method ours-without-roll
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

log = get_logger('preprocess.ours_without_roll.eve')

BATCH_SIZE = 50


def run(config, recorder: FailureRecorder):
    v2_dir = Path(config.v2_dir)
    raw = Path(config.raw_data_dir)
    out_dir = Path(config.output_dir)
    sp = config.splits                     # yaml_to_ns 后为 namespace，转 dict 统一
    splits = sp if isinstance(sp, dict) else vars(sp)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info('v3 EVE: fixed_forward=True (roll-only), world = basler cam')

    for split, subjects in splits.items():
        sel = list(subjects or [])
        override = getattr(config, 'subjects', None)
        if override:
            sel = [s for s in sel if s in override]   # 与本 split 名单求交
        for subj in tqdm(sel, desc=split, unit='subj'):
            _subject(recorder, v2_dir / split, raw, out_dir / split,
                     split, subj, int(getattr(config, 'max_frames', 0) or 0))


def _step_calib(raw, subj, step, cameras):
    """该 step 各相机 (K, ext)：ext = R_basler·R_c^T（世界系 = basler 系）"""
    d = raw / subj / step
    with h5py.File(d / 'basler.h5', 'r') as fb:
        R_b = np.array(fb['camera_transformation'], dtype=float)[:3, :3]
    cal = {}
    for cam_name in cameras:
        with h5py.File(d / f'{cam_name}.h5', 'r') as fc:
            K = np.array(fc['camera_matrix'], dtype=float)
            R_c = np.array(fc['camera_transformation'], dtype=float)[:3, :3]
        cal[cam_name] = (K, R_b @ R_c.T)
    return cal


def _subject(recorder, v2_dir, raw, out_dir, split, subj, max_rows):
    src = v2_dir / f'{subj}.h5'
    if not src.is_file():
        recorder.add(subj, '<subject>', 'no_v2_product')
        return
    with h5py.File(src, 'r') as f:
        fr = f['frame_index'][:].ravel()
        ci = f['cam_index'][:].ravel()
        st = f['step_index'][:].ravel()
        gaze = f['face_gaze'][:]
        lm3d = f['face_landmarks_3d'][:]
        lm2d = f['facial_landmarks_2d'][:]
        matn = f['face_mat_norm'][:]
        model = np.array(f.attrs['face_model'])
        mtype = str(f.attrs['face_model_type'])
        cameras = json.loads(f.attrs['cameras'])
        steps = json.loads(f.attrs['steps'])

    order = list(range(len(fr)))[:max_rows or None]
    by_step = {}
    for r in order:
        by_step.setdefault(int(st[r]), []).append(r)      # 保持 v2 存储顺序

    out_dir.mkdir(parents=True, exist_ok=True)
    n_est = len(order)
    out = h5py.File(out_dir / f'{subj}.h5', 'w')
    dsets = {
        'frame_index': out.create_dataset('frame_index', (n_est,), np.int32,
                                          chunks=(BATCH_SIZE,), maxshape=(None,)),
        'cam_index': out.create_dataset('cam_index', (n_est,), np.int32,
                                        chunks=(BATCH_SIZE,), maxshape=(None,)),
        'step_index': out.create_dataset('step_index', (n_est,), np.int32,
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

    pbar = tqdm(by_step.items(), desc=subj, unit='step', leave=False, mininterval=5)
    for si, rows in pbar:
        step = steps[si]
        try:
            cal = _step_calib(raw, subj, step, cameras)
        except (OSError, KeyError) as e:
            recorder.add(subj, step, f'error:{type(e).__name__}')
            continue
        caps, cur_pos = {}, {}
        for cam_name in cameras:
            mp4 = raw / subj / step / f'{cam_name}.mp4'
            if mp4.is_file():
                caps[cam_name] = cv2.VideoCapture(str(mp4))
                cur_pos[cam_name] = 0

        def _read_at(cam_name, target):
            """顺序读帧到 target（cur_pos = 下一个待读帧位置，从 0 起）"""
            cap = caps[cam_name]
            if target < cur_pos[cam_name]:            # 回退：必须 seek
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ok, img = cap.read()
                cur_pos[cam_name] = target + 1
                return img if ok else None
            while cur_pos[cam_name] < target:         # 前进：丢弃中间帧
                ok, _ = cap.read()
                if not ok:
                    return None
                cur_pos[cam_name] += 1
            ok, img = cap.read()                       # 读 target 帧
            cur_pos[cam_name] += 1
            return img if ok else None

        try:
            for r in rows:
                c, raw_f = int(ci[r]), int(fr[r])
                cam_name = cameras[c]
                if cam_name not in caps:
                    recorder.add(f'{subj}/{step}/{cam_name}',
                                 f'frame{raw_f:05d}', 'no_video')
                    continue
                img = _read_at(cam_name, raw_f)
                if img is None:
                    recorder.add(f'{subj}/{step}/{cam_name}',
                                 f'frame{raw_f:05d}', 'decode_failed')
                    continue
                K, ext = cal[cam_name]
                res = v3_core.v3_sample(img, model, lm3d[r], matn[r],
                                        gaze[r], K, ext)
                batch['frame_index'].append(raw_f)
                batch['cam_index'].append(c)
                batch['step_index'].append(si)
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
        finally:
            for cap in caps.values():
                cap.release()
        pbar.set_postfix({'written': written})
    pbar.close()
    _flush()
    for k in dsets:
        dsets[k].resize((written,) + dsets[k].shape[1:])
    out.attrs['face_model'] = model
    out.attrs['face_model_type'] = mtype
    out.attrs['pipeline'] = 'ours-without-roll v3'
    out.attrs['steps'] = json.dumps(steps)
    out.attrs['cameras'] = json.dumps(cameras)
    out.close()
    log.info(f'{subj}: {written}/{n_est} 样本')
