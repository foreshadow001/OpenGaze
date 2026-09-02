"""XGaze 3D 位置与头姿(HCS gaze) 跨相机一致性（2026-08-30，对齐 EVE 版）

三臂（全部 vs cam00 固定参考；前 10 台相机，官方外参）：
- gen6：逐相机独立 PnP（v1 形态）→ 3D 位置 + HCS；
- true6（5+5）：偶/奇两组独立 DLT → Kabsch(true6) → 组间 3D 一致性 + HCS；
- true6_full（10 台一组）：共享 DLT 头姿（v2 部署形态）→ 仅 HCS
  （xgaze 逐相机注释精确，共享头姿下 HCS = 标注+外参一致性 ~0.001°）。

中间值缓存 pos_hcs_cache.npz（逐对原始值 + 相机/被试标签 + 定位二元组
（被试名序号, frame_index），直接回溯原始数据）；命中跳过采样，
POS_HCS_REFRESH=1 强制重采。xgaze 标注为逐相机精确值（无 EVE 式插值
分发），无需 PoG 离散门控。

输出（本目录，无 md）:
  consistency_overall.csv      逐被试中位（末行 AVG）
  consistency_per_camera.csv   逐相机 vs cam00 中位（末行 AVG）
  pos_hcs_dist.png             三子图纵排：HCS(5+5)° / HCS(10台)° / 3D(5+5)mm
用法（仓库根目录）:
  /ssd/conda/envs/yanglinxuan/opengaze/bin/python \
  preprocess/zhang2015-specific-face-model/get_face_model/xgaze/metrics/frame_consistency/pos_hcs_consistency.py
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')

import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[5]
sys.path.insert(0, str(HERE.parents[2]))                    # get_face_model/
sys.path.insert(0, str(PROJECT_ROOT))

import face_model_core as core                              # noqa: E402
from utils.logger import get_logger                         # noqa: E402
from utils.normalization import estimateHeadPose, normalizeData_face  # noqa: E402

log = get_logger('preprocess.specific_face_model.xgaze.pos_hcs_consistency')

LM_DIR = '/media/yanglinxuan/ylx/xgaze_insightface_224'
ANN_DIR = '/media/yanglinxuan/Expansion/xgaze_raw/data/annotation_train'
CALIB_DIR = '/media/yanglinxuan/Expansion/xgaze_raw/calibration/cam_calibration'
FM_DIR = Path('/media/yanglinxuan/sfm/xgaze_specific_face_model/face_models')
CAMS = list(range(10))
N_FRAMES = 120
DUMMY = np.zeros((32, 32, 3), np.uint8)
REF_CAM = 0
GROUP_A = [0, 2, 4, 6, 8]
GROUP_B = [1, 3, 5, 7, 9]
CACHE = HERE / 'pos_hcs_cache.npz'


def sample():
    """逐对采样：值 + 相机/被试标签 + 定位二元组（被试序号, frame_index）"""
    GEN6 = np.loadtxt(
        PROJECT_ROOT / 'preprocess/zhang2015-insightface/face_model_xgaze.txt'
    )[[20, 23, 26, 29, 15, 19], :]
    KS, DIST, ROT, TR = {}, {}, {}, {}
    for c in CAMS:
        fs = cv2.FileStorage(str(Path(CALIB_DIR) / f'cam{c:02d}.xml'),
                             cv2.FILE_STORAGE_READ)
        KS[c] = fs.getNode('Camera_Matrix').mat()
        DIST[c] = fs.getNode('Distortion_Coefficients').mat()
        ROT[c] = fs.getNode('cam_rotation').mat()
        TR[c] = fs.getNode('cam_translation').mat().reshape(3, 1)
        fs.release()

    D = {k: [] for k in ('Pg', 'Hg', 'Pt', 'Ht', 'Hf')}
    TAG = {k: [] for k in ('cg', 'ct', 'cf', 'sg', 'st', 'sf',
                           'fg', 'ft', 'ff')}
    n_frames = []
    sids = sorted(p.stem for p in Path(LM_DIR).glob('subject*.h5'))
    for sid in tqdm(sids, desc='subjects', unit='subj'):
        si = len(n_frames)
        true6 = np.loadtxt(FM_DIR / sid / 'true6.txt')
        ann = {}
        with open(Path(ANN_DIR) / f'{sid}.csv') as f:
            for line in f:
                p = line.strip().split(',')
                ann[(p[0], p[1])] = np.array(
                    [float(p[4]), float(p[5]), float(p[6])]).reshape(3, 1)
        with h5py.File(Path(LM_DIR) / f'{sid}.h5', 'r') as f:
            fr_all = f['frame_index'][:].ravel()
            cam_all = f['cam_index'][:].ravel()
            lm_all = f['facial_landmarks_2d'][:]
        by_frame = {}
        for r in range(len(fr_all)):
            c = int(cam_all[r])
            if c in CAMS:
                by_frame.setdefault(int(fr_all[r]), []).append((c, r))
        frames = sorted(by_frame)
        if not frames:
            continue
        idx = np.linspace(0, len(frames) - 1,
                          min(N_FRAMES, len(frames))).astype(int)
        n_used = 0

        for fidx in (frames[i] for i in idx):
            rows = by_frame[fidx]
            # ---- gen6 臂：逐相机 PnP → 世界系 3D + HCS ----
            model = GEN6
            Xws, HCSg = {}, {}
            for c, r in rows:
                gp = ann.get((f'frame{fidx:04d}', f'cam{c:02d}.JPG'))
                if gp is None:
                    continue
                try:
                    rvec, tvec = estimateHeadPose(
                        lm_all[r][core.IDX6].reshape(6, 1, 2).astype(float),
                        model, KS[c], DIST[c])
                except cv2.error:
                    continue
                hR = cv2.Rodrigues(rvec)[0]
                Xws[c] = (ROT[c].T @ (hR @ model.T + tvec.reshape(3, 1)
                                       - TR[c])).T
                _, hr, gc = normalizeData_face(
                    DUMMY, model, rvec, tvec, gp, KS[c],
                    fixed_forward=False)[:3]
                HCSg[c] = (cv2.Rodrigues(hr)[0].T @ gc).ravel()
            if REF_CAM in Xws:
                for c in sorted(Xws):
                    if c == REF_CAM:
                        continue
                    D['Pg'].append(float(np.linalg.norm(
                        Xws[c] - Xws[REF_CAM], axis=1).mean()))
                    cos = float(np.clip(
                        HCSg[c] @ HCSg[REF_CAM]
                        / (np.linalg.norm(HCSg[c]) * np.linalg.norm(HCSg[REF_CAM])),
                        -1, 1))
                    D['Hg'].append(float(np.degrees(np.arccos(cos))))
                    TAG['cg'].append(c)
                    TAG['sg'].append(si)
                    TAG['fg'].append(fidx)

            # ---- true6 (5+5)：偶/奇两组独立 DLT → Kabsch(true6) ----
            model = true6
            Xg, HCSg2 = {}, {}
            for g, cams_g in (('A', GROUP_A), ('B', GROUP_B)):
                rays, pv = [], []
                for c, r in rows:
                    if c not in cams_g:
                        continue
                    lm_n = cv2.undistortPoints(
                        lm_all[r][core.IDX6].astype(np.float64).reshape(-1, 1, 2),
                        KS[c], DIST[c]).reshape(-1, 2)
                    rays.append(lm_n)
                    pv.append(np.concatenate(
                        [cv2.Rodrigues(ROT[c])[0].ravel(), TR[c].ravel()]))
                if len(rays) < 3:
                    continue
                X = core.triangulate(np.stack(rays), np.stack(pv), n_points=6)
                Xg[g] = X
                R_h, t_h = core.kabsch(model, X)
                for c, r in rows:
                    if c not in cams_g:
                        continue
                    gp = ann.get((f'frame{fidx:04d}', f'cam{c:02d}.JPG'))
                    if gp is None:
                        continue
                    rvec = cv2.Rodrigues(ROT[c] @ R_h)[0]
                    tvec = ROT[c] @ t_h.reshape(3, 1) + TR[c]
                    _, hr, gc = normalizeData_face(
                        DUMMY, model, rvec, tvec, gp, KS[c],
                        fixed_forward=False)[:3]
                    HCSg2[c] = (cv2.Rodrigues(hr)[0].T @ gc).ravel()
            if 'A' in Xg and 'B' in Xg:
                d_pos = float(np.linalg.norm(
                    Xg['A'] - Xg['B'], axis=1).mean())
                n_used += 1
                if REF_CAM in HCSg2:
                    for c in sorted(HCSg2):
                        if c == REF_CAM:
                            continue
                        cos = float(np.clip(
                            HCSg2[c] @ HCSg2[REF_CAM]
                            / (np.linalg.norm(HCSg2[c])
                               * np.linalg.norm(HCSg2[REF_CAM])), -1, 1))
                        D['Pt'].append(d_pos)
                        D['Ht'].append(float(np.degrees(np.arccos(cos))))
                        TAG['ct'].append(c)
                        TAG['st'].append(si)
                        TAG['ft'].append(fidx)

            # ---- true6_full：10 台一组 DLT 共享头姿（v2 部署形态）----
            rays, pv = [], []
            for c, r in rows:
                lm_n = cv2.undistortPoints(
                    lm_all[r][core.IDX6].astype(np.float64).reshape(-1, 1, 2),
                    KS[c], DIST[c]).reshape(-1, 2)
                rays.append(lm_n)
                pv.append(np.concatenate(
                    [cv2.Rodrigues(ROT[c])[0].ravel(), TR[c].ravel()]))
            if len(rays) >= 3:
                X = core.triangulate(np.stack(rays), np.stack(pv), n_points=6)
                R_h, t_h = core.kabsch(true6, X)
                Hf = {}
                for c, r in rows:
                    gp = ann.get((f'frame{fidx:04d}', f'cam{c:02d}.JPG'))
                    if gp is None:
                        continue
                    rvec = cv2.Rodrigues(ROT[c] @ R_h)[0]
                    tvec = ROT[c] @ t_h.reshape(3, 1) + TR[c]
                    _, hr, gc = normalizeData_face(
                        DUMMY, true6, rvec, tvec, gp, KS[c],
                        fixed_forward=False)[:3]
                    Hf[c] = (cv2.Rodrigues(hr)[0].T @ gc).ravel()
                if REF_CAM in Hf:
                    for c in sorted(Hf):
                        if c == REF_CAM:
                            continue
                        cos = float(np.clip(
                            Hf[c] @ Hf[REF_CAM]
                            / (np.linalg.norm(Hf[c]) * np.linalg.norm(Hf[REF_CAM])),
                            -1, 1))
                        D['Hf'].append(float(np.degrees(np.arccos(cos))))
                        TAG['cf'].append(c)
                        TAG['sf'].append(si)
                        TAG['ff'].append(fidx)
        n_frames.append((sid, n_used))

    np.savez(CACHE,
             subj_names=np.array([n for n, _ in n_frames]),
             n_frames=np.array([n for _, n in n_frames]),
             **{k: np.array(v) for k, v in D.items()},
             **{f'tag_{k}': np.array(v) for k, v in TAG.items()})
    log.info(f'缓存写入 {CACHE.name}（Ht {len(D["Ht"]):,} 对）')


def _dist_axis(ax, S, unit, label, color='#3b7dd8'):
    """单个分布子图（风格同 EVE hcs_true6_dist）"""
    S = np.asarray(S, dtype=float)
    hi = float(np.percentile(S, 99.5) * 2)
    lo = max(1e-4, S[S > 0].min() * 0.6)
    bins = np.logspace(np.log10(lo), np.log10(hi), 120)
    cnt, edges = np.histogram(S, bins=bins)
    dens = cnt / (len(S) * np.diff(np.log10(bins)))
    centers = np.sqrt(edges[:-1] * edges[1:])
    ax.plot(centers, dens, lw=1.8, color=color)
    ax.fill_between(centers, dens, alpha=0.15, color=color)
    ax.set_xscale('log')
    stats = {q: float(np.percentile(S, q)) for q in (50, 90, 95, 98, 99)}
    marks = [(f'p{q}={v:.3f}{unit}', v, c, '--')
             for (q, v), c in zip(sorted(stats.items()),
                                  ('tab:green', 'tab:orange', 'tab:red',
                                   'tab:purple', 'tab:brown'))]
    marks.append((f'mean={S.mean():.3f}{unit}', S.mean(), 'k', ':'))
    top = ax.get_ylim()[1]
    marks.sort(key=lambda m: m[1])
    ys = [top * 0.97, top * 0.84, top * 0.71, top * 0.58]   # 每 4 个一错开
    for i, (lbl, v, c, ls) in enumerate(marks):
        ax.axvline(v, color=c, ls=ls, lw=1.6)
        ax.text(v * 1.08, ys[i % 4], lbl, color=c, fontsize=9,
                ha='left', va='top', rotation=0)
    top10 = np.sort(S)[-10:][::-1]
    txt = f'Top-10 ({unit}):\n' + '\n'.join(
        f'{i + 1:2d}.  {v:.3f}' for i, v in enumerate(top10))
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, ha='left', va='top',
            fontsize=9, color='darkred', family='monospace',
            bbox=dict(fc='white', ec='darkred', alpha=0.85, pad=3))
    ax.set_xlim(lo, hi)
    ticks = [v for v in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5,
                         10, 20, 50)
             if lo <= v <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{v:g}' for v in ticks])
    ax.minorticks_off()
    ax.set_xlabel(f'{label} (log scale)')
    ax.set_ylabel('density')
    ax.grid(alpha=0.3)
    return stats


def main():
    if CACHE.is_file() and not os.environ.get('POS_HCS_REFRESH'):
        z = np.load(CACHE, allow_pickle=False)
        log.info(f'缓存命中 {CACHE.name}')
    else:
        sample()
        z = np.load(CACHE)
    names, nf = list(z['subj_names']), z['n_frames']
    Pg, Hg, Pt, Ht, Hf = (z['Pg'], z['Hg'], z['Pt'], z['Ht'], z['Hf'])
    tg = {k[4:]: z[k] for k in z.files if k.startswith('tag_')}

    # ---- CSV：逐被试 ----
    overall = HERE / 'consistency_overall.csv'
    rows = []
    for si, (name, n) in enumerate(zip(names, nf)):
        mg, mt, mf = (tg['sg'] == si, tg['st'] == si, tg['sf'] == si)
        if not (mg.any() and mt.any() and mf.any()):
            continue
        rows.append((name, int(n), np.median(Pg[mg]), np.median(Hg[mg]),
                     np.median(Pt[mt]), np.median(Ht[mt]),
                     np.median(Hf[mf])))
    with open(overall, 'w') as f:
        f.write('subject,n_frames,pos3d_gen6_mm,hcs_gen6_deg,'
                'pos3d_true6_mm,hcs_true6_deg,hcs_true6_full_deg\n')
        for row in rows:
            f.write(','.join(f'{v:.3f}' if isinstance(v, float) else str(v)
                             for v in row) + '\n')
        arr = np.array([r[2:] for r in rows])
        f.write(f'AVG,{np.mean([r[1] for r in rows]):.1f},'
                + ','.join(f'{v:.3f}' for v in arr.mean(0)) + '\n')

    # ---- CSV：逐相机 ----
    per_cam = HERE / 'consistency_per_camera.csv'
    with open(per_cam, 'w') as f:
        f.write('cam,n_pairs,pos3d_gen6_mm,hcs_gen6_deg,'
                'pos3d_true6_mm,hcs_true6_deg,hcs_true6_full_deg\n')
        rows_c = []
        for c in sorted(set(tg['ct'].tolist())):
            if c == REF_CAM:
                continue
            mg, mt, mf = (tg['cg'] == c, tg['ct'] == c, tg['cf'] == c)
            rows_c.append((c, int(mt.sum()), np.median(Pg[mg]),
                           np.median(Hg[mg]), np.median(Pt[mt]),
                           np.median(Ht[mt]), np.median(Hf[mf])))
        for c, n, a, b, d, e, g in rows_c:
            f.write(f'cam{c:02d},{n},{a:.3f},{b:.3f},{d:.3f},{e:.3f},'
                    f'{g:.3f}\n')
        arr_c = np.array([r[2:] for r in rows_c])
        f.write('AVG vs cam00,' + str(rows_c[0][1]) + ','
                + ','.join(f'{v:.3f}' for v in arr_c.mean(0)) + '\n')

    # ---- 双子图分布：上 HCS / 下 3D ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax1, ax1b, ax2) = plt.subplots(3, 1, figsize=(9, 11.5))
    s1 = _dist_axis(ax1, Ht, '°',
                    'true6(5+5) HCS consistency vs cam00')
    ax1.set_title(f'XGaze true6(5+5 DLT) HCS consistency  '
                  f'(n={len(Ht):,} pairs, {len(rows)} subjects)')
    s1b = _dist_axis(ax1b, Hf, '°',
                     'true6_full(10-cam) HCS consistency vs cam00',
                     color='#d08700')
    ax1b.set_title(f'XGaze true6_full(10-cam shared pose) HCS consistency  '
                   f'(n={len(Hf):,} pairs)')
    s2 = _dist_axis(ax2, Pt, 'mm',
                    'true6(5+5) 3D position consistency (group A vs B)')
    ax2.set_title(f'XGaze true6(5+5 DLT) 3D position consistency  '
                  f'(n={len(Pt):,} pairs)')
    fig.tight_layout(h_pad=2.0)
    png = HERE / 'pos_hcs_dist.png'
    fig.savefig(png, dpi=250)
    log.info(f'参考相机 cam00 | 输出 {overall.name} / {per_cam.name} / '
             f'{png.name}（{len(rows)} 被试）| gen6: 3D '
             f'{np.median([r[2] for r in rows]):.2f} mm, '
             f'HCS {np.median([r[3] for r in rows]):.2f}° | '
             f'true6(5+5): 3D {np.median([r[4] for r in rows]):.2f} mm, '
             f'HCS {np.median([r[5] for r in rows]):.2f}° '
             f'(p95 {s1[95]:.3f}°, 3D p95 {s2[95]:.2f}mm) | '
             f'true6_full: HCS {np.median([r[6] for r in rows]):.4f}°')


if __name__ == '__main__':
    main()
