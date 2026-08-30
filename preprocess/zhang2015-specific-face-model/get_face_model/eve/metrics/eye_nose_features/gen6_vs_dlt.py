"""EVE gen6 vs DLT 真值 6 指标对比（2026-08-30 自 true_face_model.py 抽出）

复算逐被试帧级 DLT（≥3 相机同步组，官方 camera_transformation 外参），
每帧 6 点距离指标 → 帧间中位，与 gen6 参考值对比：
外/内眼距、左/右眼宽、鼻宽、眼心鼻心距。

输出: 本目录 gen6_vs_dlt.png（6 子图折线；汇总进日志，无 csv/md）
用法（仓库根目录）:
  /ssd/conda/envs/yanglinxuan/opengaze/bin/python \
  preprocess/zhang2015-specific-face-model/get_face_model/eve/metrics/eye_nose_features/gen6_vs_dlt.py
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')

import sys
import json
import argparse
from multiprocessing import Pool
from pathlib import Path

import cv2
import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[5]
sys.path.insert(0, str(HERE.parents[2]))                    # get_face_model/
sys.path.insert(0, str(PROJECT_ROOT))

import face_model_core as core                              # noqa: E402
from utils.logger import get_logger                         # noqa: E402

log = get_logger('preprocess.specific_face_model.eve.gen6_vs_dlt')

LM_ROOT = '/media/yanglinxuan/zyx/EVE_dataset/eve_dataset/landmarks'
RAW_ROOT = '/media/yanglinxuan/zyx/EVE_dataset/eve_dataset'
IDX6 = core.IDX6
GEN6 = None


def init_worker():
    global GEN6
    GEN6 = np.loadtxt(
        PROJECT_ROOT / 'preprocess/zhang2015-insightface/face_model_xgaze.txt'
    )[[20, 23, 26, 29, 15, 19], :]


def process_subject(args):
    split, subject = args
    with h5py.File(Path(LM_ROOT) / split / f'{subject}.h5', 'r') as f:
        fr_all = f['frame_index'][:].ravel()
        ci_all = f['cam_index'][:].ravel()
        lm_all = f['facial_landmarks_2d'][:]
        cameras = json.loads(f.attrs['cameras'])
        steps = json.loads(f.attrs['steps'])
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
    # 帧同步：basler(cam0) 帧号 //2 对齐 webcam
    sync_map = {}
    for r in range(len(fr_all)):
        c, raw_f = int(ci_all[r]), int(fr_all[r])
        sync_map.setdefault(raw_f // 2 if c == 0 else raw_f, []).append((c, r))

    per_frame = {k: [] for k in ('外眼距IOD', '内眼距', '左眼宽', '右眼宽',
                                 '鼻宽', '眼心鼻心距')}
    for sync_f, rows in sorted(sync_map.items()):
        if len(set(c for c, _ in rows)) < 3:
            continue
        seen = {}
        for c, r in rows:
            seen.setdefault(c, r)
        rays, pv = [], []
        for c, r in seen.items():
            if c not in Ks:
                continue
            lm_n = cv2.undistortPoints(
                lm_all[r][IDX6].astype(np.float64).reshape(-1, 1, 2),
                Ks[c], None).reshape(-1, 2)
            rays.append(lm_n)
            pv.append(np.concatenate([cv2.Rodrigues(Rs[c])[0].ravel(),
                                      ts[c].ravel()]))
        if len(rays) < 3:
            continue
        X = core.triangulate(np.stack(rays), np.stack(pv), n_points=6)
        per_frame['外眼距IOD'].append(np.linalg.norm(X[0] - X[3]))
        per_frame['内眼距'].append(np.linalg.norm(X[1] - X[2]))
        per_frame['左眼宽'].append(np.linalg.norm(X[0] - X[1]))
        per_frame['右眼宽'].append(np.linalg.norm(X[2] - X[3]))
        per_frame['鼻宽'].append(np.linalg.norm(X[4] - X[5]))
        per_frame['眼心鼻心距'].append(
            np.linalg.norm(X[:4].mean(0) - X[4:].mean(0)))
    row = {k: float(np.median(v)) if v else float('nan') for k, v in per_frame.items()}
    row['subject'] = subject
    return row


def main():
    parser = argparse.ArgumentParser(description='EVE gen6 vs DLT 6 指标')
    parser.add_argument('-j', '--jobs', type=int, default=12)
    args = parser.parse_args()

    init_worker()
    subjects = [(sp, p.stem) for sp in ('train', 'test')
                for p in sorted(Path(LM_ROOT, sp).glob('*.h5'))]
    with Pool(args.jobs, initializer=init_worker) as pool:
        rows = sorted(pool.imap_unordered(process_subject, subjects),
                      key=lambda r: r['subject'])

    GEN6_REF = {
        '外眼距IOD': float(np.linalg.norm(GEN6[0] - GEN6[3])),
        '内眼距': float(np.linalg.norm(GEN6[1] - GEN6[2])),
        '左眼宽': float(np.linalg.norm(GEN6[0] - GEN6[1])),
        '右眼宽': float(np.linalg.norm(GEN6[2] - GEN6[3])),
        '鼻宽': float(np.linalg.norm(GEN6[4] - GEN6[5])),
        '眼心鼻心距': float(np.linalg.norm(GEN6[:4].mean(0) - GEN6[4:].mean(0))),
    }
    for k, g in GEN6_REF.items():            # 汇总进日志
        v = np.array([r[k] for r in rows])
        log.info('{}: gen6 {:.1f} | DLT {:.1f}±{:.1f} 中位 {:.1f} '
                 '[{:.1f}~{:.1f}] 差中位 {:+.1f}'.format(
                     k, g, v.mean(), v.std(), np.median(v), v.min(), v.max(),
                     float(np.median(v - g))))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    EN = {'外眼距IOD': 'Outer IOD', '内眼距': 'Inner canthal',
          '左眼宽': 'L eye width', '右眼宽': 'R eye width',
          '鼻宽': 'Nose width', '眼心鼻心距': 'Eye-nose dist'}
    order = ['外眼距IOD', '左眼宽', '鼻宽', '内眼距', '右眼宽', '眼心鼻心距']
    xs = np.arange(len(rows))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, k in zip(axes.ravel(), order):
        v = np.array([r[k] for r in rows])
        ax.plot(xs, v, '-o', ms=2.5, lw=0.9, color='tab:blue', label='DLT per-subj')
        ax.axhline(v.mean(), color='tab:green', lw=1.8, label=f'DLT mean={v.mean():.1f}')
        ax.axhline(GEN6_REF[k], color='tab:red', ls='--', lw=1.5,
                   label=f'gen6={GEN6_REF[k]:.1f}')
        ax.set_title(EN[k])
        ax.set_xlabel('subject number')
        ax.set_ylabel('mm')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    fig.suptitle(f'EVE: gen6 vs DLT (official extrinsics, 4 cams, '
                 f'{len(rows)} subjects)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = HERE / 'gen6_vs_dlt.png'
    fig.savefig(out, dpi=130)
    log.info(f'输出 {out}')


if __name__ == '__main__':
    main()
