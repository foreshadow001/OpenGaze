"""gen6 vs DLT 三角化真值的 6 点几何对比（2026-08-29，官方外参版）

以"每帧严格 DLT 三角化"的 6 点 3D 为真值代理，直接与静态 gen6 模型比几何
（gen6 列为其自身距离常数，无需 PnP；gen6+PnP 的输出是 gen6 的刚体摆放，
距离与静态模型完全相同）：
  距离类指标（逐帧计算、逐被试取中位）：
    外眼距 IOD（眼外角间距）｜内眼距（眼内角间距）｜左/右眼宽｜鼻宽（两鼻点距）
    ｜眼心—鼻心距（4 眼角均值到 2 鼻点均值的距离）

三角化配方与 new-dataset-preprocess/calculate_gaze_direction.py 一致：
官方 cam_rotation/cam_translation 原样构造 P=[R|t]（X_cam = R·X_world + t，
无任何自标），特征点用存储的翻转（正立）坐标系 h5，全部 18 台相机参与逐点 DLT；
同时记录各相机重投影残差（px）以透明化外参一致性。120 帧采样，逐帧评估。

输出: 本目录 {gen6_vs_dlt.csv, gen6_vs_dlt.png, gen6_vs_dlt.md}
用法（仓库根目录）: .../gen6_vs_dlt.py [-j 14]
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
PROJECT_ROOT = HERE.parents[5]
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(PROJECT_ROOT))

import face_model_core as core                           # noqa: E402
from utils.logger import get_logger                      # noqa: E402

log = get_logger('preprocess.specific_face_model.xgaze.gen6_vs_dlt')

LM_DIR = '/media/yanglinxuan/ylx/xgaze_insightface_224'
CALIB_DIR = '/media/yanglinxuan/Expansion/xgaze_raw/calibration/cam_calibration'
METRICS_DIR = HERE                                    # md/png → 本目录
CAM_USE = list(range(18))
IDX6 = core.IDX6
LBL6 = ['eye_out_L', 'eye_in_L', 'eye_in_R', 'eye_out_R', 'nose_L', 'nose_R']
GEN6 = None
KS = DISTS = None
ROT = TR = None

GEN6_REF = None      # gen6 距离指标（worker 初始化后填充）


def init_worker():
    global GEN6, KS, DISTS, GEN6_REF
    GEN6 = np.loadtxt(PROJECT_ROOT / 'preprocess/zhang2015-insightface/face_model_xgaze.txt')[
        [20, 23, 26, 29, 15, 19], :]
    global ROT, TR
    KS, DISTS, ROT, TR = {}, {}, {}, {}
    for c in CAM_USE:
        fs = cv2.FileStorage(str(Path(CALIB_DIR) / f'cam{c:02d}.xml'),
                             cv2.FILE_STORAGE_READ)
        KS[c] = fs.getNode('Camera_Matrix').mat()
        DISTS[c] = fs.getNode('Distortion_Coefficients').mat()
        ROT[c] = fs.getNode('cam_rotation').mat()
        TR[c] = fs.getNode('cam_translation').mat().reshape(3, 1)
        fs.release()
    g = GEN6
    GEN6_REF = {
        '外眼距IOD': np.linalg.norm(g[0] - g[3]),
        '内眼距': np.linalg.norm(g[1] - g[2]),
        '左眼宽': np.linalg.norm(g[0] - g[1]),
        '右眼宽': np.linalg.norm(g[2] - g[3]),
        '鼻宽': np.linalg.norm(g[4] - g[5]),
        '眼心鼻心距': np.linalg.norm(g[:4].mean(0) - g[4:].mean(0)),
    }


def process_subject(sid):
    subject = f'subject{sid:04d}'
    idx6_rows = [core.RIGID.index(i) for i in IDX6]
    with h5py.File(Path(LM_DIR) / f'{subject}.h5', 'r') as f:
        fr_all = f['frame_index'][:].ravel()
        cam_all = f['cam_index'][:].ravel()
        lm_all = f['facial_landmarks_2d'][:]

    frames_universe = sorted(set(fr_all.tolist()))
    sel = np.array(frames_universe)[np.linspace(
        0, len(frames_universe) - 1, min(120, len(frames_universe))).astype(int)]
    all_frames = sorted(sel.tolist())

    # 官方外参 DLT（与 new-dataset-preprocess 配方一致）：P=[R|t]，全部 18 台
    per_frame = {k: [] for k in GEN6_REF}
    res_px = []
    for f in all_frames:
        views = []
        for c in CAM_USE:
            m = np.where((cam_all == c) & (fr_all == f))[0]
            if len(m) == 0:
                continue
            lm_px = lm_all[m[0]][IDX6].astype(np.float64)
            lm_n = cv2.undistortPoints(
                lm_px.reshape(-1, 1, 2), KS[c], DISTS[c]).reshape(-1, 2)
            views.append((c, lm_n, lm_px))
        if len(views) < 6:
            continue
        lm = np.stack([v[1] for v in views])
        pv = np.stack([np.concatenate([
            cv2.Rodrigues(ROT[c])[0].ravel(), TR[c].ravel()]) for c, _, _ in views])
        X = core.triangulate(lm, pv, n_points=6)          # (6,3) 官方世界系
        for c, _, lm_px in views:
            pr, _ = cv2.projectPoints(X.reshape(-1, 1, 3),
                                      cv2.Rodrigues(ROT[c])[0], TR[c], KS[c], DISTS[c])
            res_px.append(np.linalg.norm(pr.reshape(6, 2) - lm_px, axis=1).mean())

        per_frame['外眼距IOD'].append(np.linalg.norm(X[0] - X[3]))
        per_frame['内眼距'].append(np.linalg.norm(X[1] - X[2]))
        per_frame['左眼宽'].append(np.linalg.norm(X[0] - X[1]))
        per_frame['右眼宽'].append(np.linalg.norm(X[2] - X[3]))
        per_frame['鼻宽'].append(np.linalg.norm(X[4] - X[5]))
        per_frame['眼心鼻心距'].append(np.linalg.norm(X[:4].mean(0) - X[4:].mean(0)))

    row = {'subject': subject}
    for k, v in per_frame.items():
        row[k] = float(np.median(v)) if v else float('nan')
    row['res_px'] = float(np.median(res_px)) if res_px else float('nan')
    return row


def main():
    parser = argparse.ArgumentParser(description='gen6 vs DLT 三角化真值几何对比')
    parser.add_argument('-j', '--jobs', type=int, default=14)
    args = parser.parse_args()

    init_worker()                          # 主进程也需要 GEN6_REF（聚合/绘图用）
    sids = sorted(int(p.stem.replace('subject', '')) for p in Path(LM_DIR).glob('subject*.h5'))
    log.info(f'gen6 vs DLT: {len(sids)} 被试 × 前 10 台相机 × test 帧 DLT')
    with Pool(args.jobs, initializer=init_worker) as pool:
        rows = list(pool.imap_unordered(process_subject, sids))
    rows.sort(key=lambda r: r['subject'])
    log.info('完成; 官方外参 DLT 重投影残差 中位 {:.1f}px, p90 {:.1f}px'.format(
        np.median([r['res_px'] for r in rows]),
        np.percentile([r['res_px'] for r in rows], 90)))

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    for k in GEN6_REF:                     # 汇总进日志（不落 md）
        v = np.array([r[k] for r in rows])
        g = GEN6_REF[k]
        log.info('{}: gen6 {:.1f} | DLT {:.1f}±{:.1f} 中位 {:.1f} [{:.1f}~{:.1f}] 差中位 {:+.1f}'.format(
            k, g, v.mean(), v.std(), np.median(v), v.min(), v.max(),
            float(np.median(v - g))))

    # 折线图：三列分组——外/内眼距 ｜ 左右眼宽 ｜ 鼻宽+眼心鼻心距
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    EN = {'外眼距IOD': 'Outer IOD', '内眼距': 'Inner canthal',
          '左眼宽': 'L eye width', '右眼宽': 'R eye width',
          '鼻宽': 'Nose width', '眼心鼻心距': 'Eye-nose dist'}
    # 六子图按列分组：第1列 外/内眼距，第2列 左/右眼宽，第3列 鼻宽/眼心鼻心距
    order = ['外眼距IOD', '左眼宽', '鼻宽', '内眼距', '右眼宽', '眼心鼻心距']
    xs = np.arange(len(rows))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, k in zip(axes.ravel(), order):
        v = np.array([r[k] for r in rows])
        ax.plot(xs, v, '-o', ms=2.5, lw=0.9, color='tab:blue',
                label='DLT per-subj')
        ax.axhline(v.mean(), color='tab:green', lw=1.8,
                   label=f'DLT mean = {v.mean():.1f}')
        ax.axhline(GEN6_REF[k], color='tab:red', ls='--', lw=1.5,
                   label=f'gen6 = {GEN6_REF[k]:.1f}')
        ax.set_title(EN[k])
        ax.set_xlabel('subject number')
        ax.set_ylabel('mm')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    fig.suptitle('gen6 vs DLT-triangulated 6-point geometry (80 subjects, official extrinsics, '
                 '18 cams, no filter)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(METRICS_DIR / 'gen6_vs_dlt.png', dpi=130)
    log.info(f"输出 {METRICS_DIR / 'gen6_vs_dlt.png'}")


if __name__ == '__main__':
    main()
