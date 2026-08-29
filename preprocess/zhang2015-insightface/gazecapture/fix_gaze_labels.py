"""GC yaw 标签修复：不读图、不 warp，只重算 (θ,φ) 写回 h5

原理：yaw 修复仅改 _gaze_point_cam 的 x 方向 → gaze_point 变化 → gc_normalized 变化
→ (θ,φ) 变化。face_patch / face_mat_norm / facial_landmarks_2d 完全不变
（warp 矩阵 W = f(K, S, R) 不含 gaze_point）。

对每帧：读 landmarks + orientation（从已有 h5）→ 读 dotInfo → PnP（用对应 face model）
→ face_center → R → gc_normalized = R @ (gaze_point − face_center) → 写回 face_gaze。

用法：
  python preprocess/zhang2015-insightface/gazecapture/fix_gaze_labels.py \
      [--models generic|personalized|both]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.normalization import estimateHeadPose  # noqa: E402
from utils.logger import get_logger  # noqa: E402

log = get_logger('preprocess.gazecapture.fix_labels')

IDX6 = [35, 39, 89, 93, 78, 84]
LANDMARK_USE = [20, 23, 26, 29, 15, 19]
GEN6 = None  # 延迟加载

RAW_DIR = '/media/yanglinxuan/zyx/GazeCapture'
LM_DIR = '/media/yanglinxuan/zyx/GazeCapture/landmarks'
CAL_DIR = '/media/yanglinxuan/zyx/GazeCapture/calibration'
FM_ROOT = '/media/yanglinxuan/ylx/gazecapture_specific_face_model/face_models'
SFM_SPLIT = 'configs/splits/gazecapture_sfm.yaml'

V1_DIR = '/media/yanglinxuan/ylx/gazecapture_insightface_224'
V2_DIR = '/media/yanglinxuan/sfm/gazecapture_specific_224'


def _slugify(s):
    return s.lower().replace(' ', '-')


def _dot_to_ccs_mm(ori, xcam, ycam):
    if ori == 1: return -xcam * 10, -ycam * 10
    if ori == 2: return xcam * 10, ycam * 10
    if ori == 3: return -ycam * 10, xcam * 10
    if ori == 4: return ycam * 10, -xcam * 10
    raise ValueError(ori)


def _gaze_point_cam(ori, ccs_x, ccs_y):
    """修正后的映射（2026-08-28 yaw 修复版）"""
    if ori in (1, 2):
        p = (ccs_x, ccs_y, 0.0)
    else:
        p = (-ccs_y, ccs_x, 0.0)
    if ori in (2, 4):
        p = (-p[0], -p[1], 0.0)
    return p


def load_K(device, w, h, cache):
    key = (device, w, h)
    if key not in cache:
        fs = cv2.FileStorage(str(Path(CAL_DIR) / f'{_slugify(device)}_{w}x{h}.xml'),
                             cv2.FILE_STORAGE_READ)
        cache[key] = (fs.getNode('Camera_Matrix').mat(),
                      fs.getNode('Distortion_Coefficients').mat())
        fs.release()
    return cache[key]


def fix_h5(h5_path, session, model_kind, calib_cache):
    """修复单个 h5 的 face_gaze（不读图、不 warp）"""
    global GEN6
    if GEN6 is None:
        fm = np.loadtxt(str(Path(__file__).parent.parent / 'face_model_xgaze.txt'))
        GEN6 = fm.reshape(50, 1, 3)[LANDMARK_USE, :]

    raw = Path(RAW_DIR) / session
    try:
        dot = json.load(open(raw / 'dotInfo.json'))
        device = json.load(open(raw / 'info.json'))['DeviceName']
    except Exception:
        return 0

    # 获取帧尺寸（读 1 帧）
    frames_list = json.load(open(raw / 'frames.json'))
    if not frames_list:
        return 0
    img0 = cv2.imread(str(raw / 'frames' / frames_list[0]))
    if img0 is None:
        return 0
    h_px, w_px = img0.shape[:2]
    K, dist = load_K(device, w_px, h_px, calib_cache)
    pos_of = {int(n.split('.')[0]): i for i, n in enumerate(frames_list)}

    with h5py.File(h5_path, 'r+') as h5:
        fi = h5['frame_index'][:].ravel()
        ori_arr = h5['orientation'][:].ravel()
        lm_all = h5['facial_landmarks_2d'][:]
        n = len(fi)
        pos_cache = {}
        dual_model_cache = {}

        for r in range(n):
            fr = int(fi[r])
            ori = int(ori_arr[r])
            pos = pos_of.get(fr)
            if pos is None:
                continue
            xc, yc = dot['XCam'][pos], dot['YCam'][pos]
            if dot['DotNum'][pos] == -1 or xc is None:
                continue

            # 选模型
            if model_kind == 'personalized':
                key = (session, ori)
                if key not in dual_model_cache:
                    p = Path(FM_ROOT) / session / f'ori{ori}_model6.txt'
                    dual_model_cache[key] = (np.loadtxt(str(p)).astype(float)
                                             if p.is_file() else GEN6)
                model = dual_model_cache[key]
            else:
                model = GEN6

            # PnP
            pts = lm_all[r][IDX6].reshape(6, 1, 2).astype(float)
            rvec, tvec = estimateHeadPose(pts, model, K, dist)

            # face_center + R（与 normalizeData_face 一致）
            ht = tvec.reshape(3, 1)
            hR = cv2.Rodrigues(rvec)[0]
            face = model.reshape(6, 3).T
            Fc = hR @ face + ht
            two_eye = np.mean(Fc[:, 0:4], axis=1).reshape(3, 1)
            nose = np.mean(Fc[:, 4:6], axis=1).reshape(3, 1)
            face_center = np.mean(np.concatenate((two_eye, nose), axis=1),
                                  axis=1).reshape(3, 1)
            distance = np.linalg.norm(face_center)
            hRx = hR[:, 0]
            forward = (face_center / distance).reshape(3)
            down = np.cross(forward, hRx)
            down /= np.linalg.norm(down)
            right = np.cross(down, forward)
            right /= np.linalg.norm(right)
            R = np.c_[right, down, forward].T

            # 修正后的 gaze_point
            ccs_x, ccs_y = _dot_to_ccs_mm(ori, xc, yc)
            gp = np.array(_gaze_point_cam(ori, ccs_x, ccs_y)).reshape(3, 1)

            # gc_normalized → (θ, φ)
            gc_norm = R @ (gp - face_center)
            gc_norm = gc_norm / np.linalg.norm(gc_norm)
            th = np.arcsin(-gc_norm[1, 0])
            ph = np.arctan2(-gc_norm[0, 0], -gc_norm[2, 0])

            h5['face_gaze'][r] = (th, ph)
        return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default='both', choices=['generic', 'personalized', 'both'])
    args = parser.parse_args()

    calib_cache = {}
    sfm = yaml.safe_load(open(SFM_SPLIT))
    sfm_sessions = set(sfm.get('train', []) + sfm.get('test', []))

    total = 0
    t0 = time.time()
    for split in ('train', 'test'):
        lm_dir = Path(LM_DIR) / split
        sessions = sorted(p.stem for p in lm_dir.glob('*.h5'))
        log.info(f'===== {split}: {len(sessions)} session =====')
        pbar = tqdm(sessions, desc=f'fix-{split}', unit='session', ncols=100)

        for session in pbar:
            # v1（通用模型）
            if args.models in ('generic', 'both'):
                h1 = Path(V1_DIR) / split / f'{session}.h5'
                if h1.is_file():
                    try:
                        fix_h5(h1, session, 'generic', calib_cache)
                    except Exception as e:
                        log.warning(f'{session} v1: {e}')

            # v2（个性化模型，仅 sfm 名单）
            if args.models in ('personalized', 'both') and session in sfm_sessions:
                h2 = Path(V2_DIR) / split / f'{session}.h5'
                if h2.is_file():
                    try:
                        fix_h5(h2, session, 'personalized', calib_cache)
                    except Exception as e:
                        log.warning(f'{session} v2: {e}')

            total += 1
            pbar.set_postfix({'done': total})
        pbar.close()

    log.info(f'完成: {total} session, 耗时 {(time.time() - t0) / 60:.1f} 分钟')


if __name__ == '__main__':
    main()
