"""xgaze 每人真实 6 点人脸模型（官方外参 DLT 版，2026-08-29）

第二版预处理重跑第一步：为每个被试建立"真实"6 点模型。
方法（全部官方数据，无任何自标/替代）：
  - 每帧：18 台相机的 insightface 6 点（存储于翻转/正立坐标系 h5）经官方
    K/dist 去畸变 → 官方外参 P=[R|t] 逐点 DLT → 该帧 6 点世界坐标；
  - 逐帧 Kabsch（刚体、无缩放）对齐到 gen6（仅消除头的运动，不引入 gen6 几何）；
  - 逐点取帧间中位 → 每人真实 6 点模型（gen6 朝向的坐标系）。
  - 使用该被试全部帧（200~611 帧 × 18 相机 = 3532~10993 张图）。

标准系（解剖轴定义，2026-08-29 定稿，详见 README）：
  roll=0 ⇔ 两眼中心连线 ∥ x；yaw=0 ⇔ 眼心—鼻心连线 ⊥ x；pitch=0 ⇔ 眼心—鼻心
  连线 ∥ y；x̂=眼心连线，ŷ=眼→鼻去 x̂ 分量，ẑ=x̂×ŷ（右手系），原点=眼心。

输出：
  /media/yanglinxuan/sfm/xgaze_specific_face_model/face_models/<subject>/true6.txt
      —— gen6 对齐系（逐帧 Kabsch 到 gen6 的中位）
  /media/yanglinxuan/sfm/xgaze_specific_face_model/face_models/<subject>/true6_canonical.txt
      —— 标准系（对 true6 做解剖轴标准化）
  canonical_mean6.txt —— 80 人标准系均值（标准模型）
  几何对比/逐人指标：metrics/eye_nose_features/gen6_vs_dlt.py
用法（仓库根目录）: .../true_face_model.py [-j 14]
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys
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

log = get_logger('preprocess.specific_face_model.xgaze.true_face_model')

LM_DIR = '/media/yanglinxuan/ylx/xgaze_insightface_224'
CALIB_DIR = '/media/yanglinxuan/Expansion/xgaze_raw/calibration/cam_calibration'
OUT_DIR = Path('/media/yanglinxuan/sfm/xgaze_specific_face_model/face_models')
METRICS_DIR = HERE / 'metrics' / 'true_model'
CAM_USE = list(range(18))
IDX6 = core.IDX6
GEN6 = None
KS = DISTS = ROT = TR = None


def init_worker():
    global GEN6, KS, DISTS, ROT, TR
    GEN6 = np.loadtxt(PROJECT_ROOT / 'preprocess/zhang2015-insightface/face_model_xgaze.txt')[
        [20, 23, 26, 29, 15, 19], :]
    KS, DISTS, ROT, TR = {}, {}, {}, {}
    for c in CAM_USE:
        fs = cv2.FileStorage(str(Path(CALIB_DIR) / f'cam{c:02d}.xml'),
                             cv2.FILE_STORAGE_READ)
        KS[c] = fs.getNode('Camera_Matrix').mat()
        DISTS[c] = fs.getNode('Distortion_Coefficients').mat()
        ROT[c] = fs.getNode('cam_rotation').mat()
        TR[c] = fs.getNode('cam_translation').mat().reshape(3, 1)
        fs.release()


def process_subject(sid):
    subject = f'subject{sid:04d}'
    with h5py.File(Path(LM_DIR) / f'{subject}.h5', 'r') as f:
        fr_all = f['frame_index'][:].ravel()
        cam_all = f['cam_index'][:].ravel()
        lm_all = f['facial_landmarks_2d'][:]

    by_frame = {}
    for r in range(len(fr_all)):
        by_frame.setdefault(int(fr_all[r]), []).append((int(cam_all[r]), r))

    aligned = []                          # 每帧 Kabsch 对齐到 gen6 后的 (6,3)
    n_imgs = 0
    for fidx, rows in sorted(by_frame.items()):
        rays, pv = [], []
        for c, r in rows:
            lm_px = lm_all[r][IDX6].astype(np.float64)
            lm_n = cv2.undistortPoints(
                lm_px.reshape(-1, 1, 2), KS[c], DISTS[c]).reshape(-1, 2)
            rays.append(lm_n)
            pv.append(np.concatenate([cv2.Rodrigues(ROT[c])[0].ravel(),
                                      TR[c].ravel()]))
            n_imgs += 1
        if len(rays) < 6:
            continue
        X = core.triangulate(np.stack(rays), np.stack(pv), n_points=6)
        R, t = core.kabsch(X, GEN6)        # 刚体无缩放：只消头运动
        aligned.append(((R @ X.T).T + t))
    model = np.median(np.stack(aligned), axis=0)
    model_c, _, _ = canonicalize(model)     # 解剖轴标准系

    sub = OUT_DIR / subject
    sub.mkdir(parents=True, exist_ok=True)
    np.savetxt(sub / 'true6.txt', model, fmt='%.6f')
    np.savetxt(sub / 'true6_canonical.txt', model_c, fmt='%.6f')
    return {'subject': subject, 'n_frames': len(aligned), 'n_imgs': n_imgs,
            'iod': float(np.linalg.norm(model[0] - model[3])),
            'nose_w': float(np.linalg.norm(model[4] - model[5])),
            'eye_nose': float(np.linalg.norm(model[:4].mean(0) - model[4:].mean(0)))}


def main():
    parser = argparse.ArgumentParser(description='xgaze 每人真实 6 点模型（官方外参 DLT）')
    parser.add_argument('-j', '--jobs', type=int, default=14)
    args = parser.parse_args()

    init_worker()
    sids = sorted(int(p.stem.replace('subject', '')) for p in Path(LM_DIR).glob('subject*.h5'))
    log.info(f'真实 6 点模型: {len(sids)} 被试 × 全部帧 × 18 相机（官方外参）')
    with Pool(args.jobs, initializer=init_worker) as pool:
        rows = list(pool.imap_unordered(process_subject, sids))
    rows.sort(key=lambda r: r['subject'])
    # 标准模型 = 全部被试 true6_canonical 逐点均值（原出 canonical_model.py）
    mean_c = np.mean([np.loadtxt(OUT_DIR / r['subject'] / 'true6_canonical.txt')
                      for r in rows], axis=0)
    np.savetxt(OUT_DIR / 'canonical_mean6.txt', mean_c, fmt='%.6f')
    log.info(f'标准模型写出 {OUT_DIR / "canonical_mean6.txt"}')
    log.info('模型均值: IOD {:.1f} 鼻宽 {:.1f} 眼心鼻心 {:.1f} mm（gen6: 91.3/21.9/48.7）'.format(
        np.mean([r['iod'] for r in rows]), np.mean([r['nose_w'] for r in rows]),
        np.mean([r['eye_nose'] for r in rows])))



if __name__ == '__main__':
    main()
