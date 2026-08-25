"""按设备扫描 fx 的 PnP 重投影自标定（obtain_camera_intrinsics.md §6 的独立实现）

原理：insightface 106 关键点中取 6 点（IDX6）与 3D face model 构成 3D-2D 对应，
给定候选 fx 做 solvePnP，重投影误差最小的 fx 即该设备前置真实像素焦距——
等效在线棋盘格标定，不依赖外部资料，标定的是实际采出数据的这颗镜头。

用法（在仓库根目录，conda activate opengaze）：
    python preprocess/zhang2015-insightface/gazecapture/calibrate_fx.py \
        --devices "iPhone 6" "iPhone 6s" "iPad Air 2" --per-device 60
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

GAZE_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(GAZE_ROOT))

from utils.normalization import estimateHeadPose  # noqa: E402

HERE = Path(__file__).resolve().parent
LANDMARK_USE = [20, 23, 26, 29, 15, 19]
IDX6 = [35, 39, 89, 93, 78, 84]
FACE_MODEL_USE = np.loadtxt(HERE.parent / 'face_model_xgaze.txt').reshape(50, 1, 3)[LANDMARK_USE, :]
DIST = np.zeros(5)
DATA = Path('/media/hitsz/zyx/GazeCapture')  # 2026-08 数据由 ylx 迁至 zyx，改指新位置以保持可复现


def collect_frames(device, n_sessions, per_session):
    """找该设备的 session，均匀抽帧原图（横竖混合，逐帧按实际尺寸构造 K）"""
    frames = []
    for d in sorted(os.listdir(DATA)):
        if len(frames) >= n_sessions * per_session:
            break
        if not d.isdigit():
            continue
        try:
            if json.load(open(DATA / d / 'info.json')).get('DeviceName') != device:
                continue
        except Exception:
            continue
        fdir = DATA / d / 'frames'
        files = sorted(os.listdir(fdir))
        step = max(1, len(files) // (per_session // n_sessions + 1))
        got = 0
        for f in files[::step]:
            if got >= per_session // n_sessions:
                break
            img = cv2.imread(str(fdir / f))
            if img is not None:
                frames.append((d, f, img))
                got += 1
    return frames


def reprojection_error(img, lm106, fx):
    h, w = img.shape[:2]
    K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=np.float64)
    pts2d = lm106[IDX6].reshape(6, 1, 2).astype(np.float64)
    try:
        rvec, tvec = estimateHeadPose(pts2d, FACE_MODEL_USE, K, DIST)
    except cv2.error:
        return None
    proj, _ = cv2.projectPoints(FACE_MODEL_USE, rvec, tvec, K, DIST)
    return float(np.sqrt(np.mean((proj.reshape(-1, 2) - pts2d.reshape(-1, 2)) ** 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--devices', nargs='+',
                        default=['iPhone 6', 'iPhone 6 Plus', 'iPhone 6s', 'iPad Air 2'])
    parser.add_argument('--n-sessions', type=int, default=2)
    parser.add_argument('--per-session', type=int, default=30)
    parser.add_argument('--fx-range', type=int, nargs=2, default=[450, 751])
    parser.add_argument('--step', type=int, default=10)
    args = parser.parse_args()

    from insightface.app import FaceAnalysis
    app = FaceAnalysis('buffalo_l', allowed_modules=['detection', 'landmark_2d_106'],
                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0)

    for device in args.devices:
        frames = collect_frames(device, args.n_sessions, args.per_session)
        detections = []
        for sess, fname, img in frames:
            faces = app.get(img)
            if faces:
                detections.append((sess, fname, img, faces[0].landmark_2d_106))
        if len(detections) < 10:
            print(f'{device}: 有效帧不足（{len(detections)}），跳过')
            continue

        print(f'\n===== {device}: {len(detections)} 帧检测成功 =====')
        print(f'{"fx":>6}{"重投影RMS(px)":>14}   谷底标记')
        best = (None, 1e9)
        curve = []
        for fx in range(args.fx_range[0], args.fx_range[1], args.step):
            errs = [e for _, _, img, lm in detections
                    if (e := reprojection_error(img, lm, fx)) is not None]
            rms = float(np.mean(errs)) if errs else float('nan')
            curve.append((fx, rms))
            if rms < best[1]:
                best = (fx, rms)
        for fx, rms in curve:
            mark = '  ← 最小' if fx == best[0] else ''
            print(f'{fx:>6}{rms:>14.3f}{mark}')
        print(f'{device} 最优 fx = {best[0]}（RMS {best[1]:.3f}px）')


if __name__ == '__main__':
    main()
