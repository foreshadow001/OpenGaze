"""EVE 每人真实 6 点人脸模型（官方外参 DLT 版，2026-08-29）

与 xgaze 版（../xgaze/true_face_model.py）同方法，差异：
  - 4 台相机（basler + webcam_l/c/r），官方外参 camera_transformation（4x4，
    Xc = R·Xw + t，跨 step 恒定已验证），内参逐相机 h5 camera_matrix，无畸变
  - 特征点在翻转/正立坐标系 h5（landmarks/<split>/<被试>.h5）
  - 组门控：四相机齐全且 PoG validity 全真（缺一即弃，2026-08-30 定稿）

标准系与 xgaze 版一致（解剖轴定义，见 CLAUDE.md 约定 9）。

输出：
  /media/yanglinxuan/sfm/eve_specific_face_model/face_models/<被试>/
    true6.txt (gen6 对齐系) / true6_canonical.txt (标准系)
  canonical_mean6.txt —— 全部被试 true6_canonical 均值（标准模型）
  （gen6 vs DLT 6 指标对比：独立脚本
    metrics/eye_nose_features/gen6_vs_dlt.py）
用法：.../true_face_model.py [-j 12]
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys
import json
import argparse
from pathlib import Path
from multiprocessing import Pool

import cv2
import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(PROJECT_ROOT))

import face_model_core as core                           # noqa: E402
from utils.logger import get_logger                      # noqa: E402
from utils.normalization import canonicalize_face_model as canonicalize  # noqa: E402,F401

log = get_logger('preprocess.specific_face_model.eve.true_face_model')

LM_ROOT = '/media/yanglinxuan/zyx/EVE_dataset/eve_dataset/landmarks'
RAW_ROOT = '/media/yanglinxuan/zyx/EVE_dataset/eve_dataset'
OUT_DIR = Path('/media/yanglinxuan/sfm/eve_specific_face_model/face_models')
IDX6 = core.IDX6
GEN6 = None
IDX6_ROWS = None


def init_worker():
    global GEN6, IDX6_ROWS
    GEN6 = np.loadtxt(PROJECT_ROOT / 'preprocess/zhang2015-insightface/face_model_xgaze.txt')[
        [20, 23, 26, 29, 15, 19], :]
    IDX6_ROWS = [core.RIGID.index(i) for i in IDX6]


def process_subject(args):
    split, subject = args
    lm_path = Path(LM_ROOT) / split / f'{subject}.h5'
    with h5py.File(lm_path, 'r') as f:
        fr_all = f['frame_index'][:].ravel()
        ci_all = f['cam_index'][:].ravel()
        st_all = f['step_index'][:].ravel()
        lm_all = f['facial_landmarks_2d'][:]
        cameras = json.loads(f.attrs['cameras'])
        steps = json.loads(f.attrs['steps'])

    # 官方内参 + 外参（跨 step 恒定已验证，取第一个在盘 step）
    Ks, Rs, ts = {}, {}, {}
    for c, cam_name in enumerate(cameras):
        for step in steps:
            p = Path(RAW_ROOT) / subject / step / f'{cam_name}.h5'
            if p.is_file():
                with h5py.File(p, 'r') as f:
                    Ks[c] = np.array(f['camera_matrix'], dtype=float)
                    T = np.array(f['camera_transformation'], dtype=float)
                Rs[c] = T[:3, :3]
                ts[c] = T[:3, 3].reshape(3, 1)
                break

    # 跨相机帧同步：basler(cam0) 30fps / webcam 15fps → basler帧号/2 = 同步帧号
    # 组门控（2026-08-30 定稿）：四相机齐全且各自 PoG validity 全真，缺一即弃
    sync_map = {}         # (sync_frame, step) -> [(cam, row, raw_f)]
    for r in range(len(fr_all)):
        c, raw_f, si = int(ci_all[r]), int(fr_all[r]), int(st_all[r])
        sync_f = raw_f // 2 if c == 0 else raw_f
        sync_map.setdefault((sync_f, si), []).append((c, r, raw_f))
    valid_cache = {}      # (step, cam) -> PoG validity 数组（懒加载）
    def _valid(si, c, raw_f):
        if (si, c) not in valid_cache:
            p = Path(RAW_ROOT) / subject / steps[si] / f'{cameras[c]}.h5'
            valid_cache[(si, c)] = None if not p.is_file() else \
                np.asarray(h5py.File(p, 'r')['face_PoG_tobii/validity'])
        v = valid_cache[(si, c)]
        return v is not None and raw_f < len(v) and bool(v[raw_f])

    aligned = []
    n_imgs = 0
    for (sync_f, si), rows in sorted(sync_map.items()):
        if set(c for c, _, _ in rows) != set(range(4)):
            continue                      # 相机不齐，弃
        if not all(_valid(si, c, raw_f) for c, _, raw_f in rows):
            continue                      # 有相机 PoG 无效，弃
        # 每相机只取一行（去重：同一相机可能有多行）
        seen = {}
        for c, r, raw_f in rows:
            if c not in seen:
                seen[c] = r
        rays, pv = [], []
        for c, r in seen.items():
            if c not in Ks:
                continue
            lm_px = lm_all[r][IDX6].astype(np.float64)
            lm_n = cv2.undistortPoints(
                lm_px.reshape(-1, 1, 2), Ks[c], None).reshape(-1, 2)
            rays.append(lm_n)
            pv.append(np.concatenate([cv2.Rodrigues(Rs[c])[0].ravel(), ts[c].ravel()]))
            n_imgs += 1
        if len(rays) < 3:
            continue
        X = core.triangulate(np.stack(rays), np.stack(pv), n_points=6)
        R, t = core.kabsch(X, GEN6)
        aligned.append(((R @ X.T).T + t))
    model = np.median(np.stack(aligned), axis=0)
    model_c, _, _ = canonicalize(model)

    sub = OUT_DIR / subject
    sub.mkdir(parents=True, exist_ok=True)
    np.savetxt(sub / 'true6.txt', model, fmt='%.6f')
    np.savetxt(sub / 'true6_canonical.txt', model_c, fmt='%.6f')

    row = {'subject': subject, 'n_frames': len(aligned), 'n_imgs': n_imgs,
           'iod_model': float(np.linalg.norm(model[0] - model[3])),
           'nose_w_model': float(np.linalg.norm(model[4] - model[5]))}
    return row


def main():
    parser = argparse.ArgumentParser(description='EVE 每人真实 6 点模型 + 一致性 + 6 指标')
    parser.add_argument('-j', '--jobs', type=int, default=12)
    args = parser.parse_args()

    init_worker()
    subjects = [(sp, p.stem) for sp in ('train', 'test')
                for p in sorted(Path(LM_ROOT, sp).glob('*.h5'))]
    log.info(f'EVE 真实模型: {len(subjects)} 被试 × 全部帧 × 4 相机（官方外参）')
    with Pool(args.jobs, initializer=init_worker) as pool:
        rows = list(pool.imap_unordered(process_subject, subjects))
    rows.sort(key=lambda r: r['subject'])
    log.info(f'建模完成: {len(rows)} 被试')

    # ---- 标准模型 ----
    models_c = np.stack([np.loadtxt(OUT_DIR / r['subject'] / 'true6_canonical.txt') for r in rows])
    mean_c = models_c.mean(axis=0)
    np.savetxt(OUT_DIR / 'canonical_mean6.txt', mean_c, fmt='%.6f')

    log.info(f'模型均值: IOD {np.mean([r["iod_model"] for r in rows]):.1f} '
             f'鼻宽 {np.mean([r["nose_w_model"] for r in rows]):.1f} mm '
             f'（{len(rows)} 被试；6 指标对比独立脚本 '
             f'metrics/eye_nose_features/gen6_vs_dlt.py）')


if __name__ == '__main__':
    main()
