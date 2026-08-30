"""EVE 3D 位置与头姿(HCS gaze) 跨相机一致性（2026-08-30，仿 xgaze 版）

组门控（2026-08-30 定稿）：四相机齐全且各自 PoG validity 全真，缺一即弃。
4 台相机无法做两组互验 DLT（2+2 基线太弱），故与 xgaze 版差异：
- gen6 臂：逐相机独立 PnP（v1 形态）→ 3D 位置（官方外参转世界系）+ HCS；
- true6 臂：**4 台一组** DLT → Kabsch(true6) 共享头姿（v2 部署形态）→ 仅 HCS
  （3D 无组间可比对象，不产出）。
固定参考相机 cam00 = basler（所有"跨相机"指标 = vs cam00）。
EVE 视线链路：官方 PoG 直算（dataset_report §4 定稿公式）——
face_PoG_tobii 屏幕像素 → ×millimeters_per_pixel → camera_transformation
→ 该相机系注视点 3D 坐标（各相机标注为同一 tobii 流按同步帧号分发，
跨相机一致 ~0.04°）。

中间值缓存 pos_hcs_cache.npz（逐对原始值 + 相机/被试标签）：
命中则跳过采样直接出 CSV/图；POS_HCS_REFRESH=1 强制重采。

输出（本目录，无 md）:
  consistency_overall.csv       逐被试中位（末行 AVG）
  consistency_per_camera.csv    逐相机 vs cam00 中位（末行 AVG）
  hcs_true6_dist.png            hcs_true6_deg 概率分布（全体对汇总，标 p50/p90/p95/p98/p99）
用法（仓库根目录）:
  /ssd/conda/envs/yanglinxuan/opengaze/bin/python \
  preprocess/zhang2015-specific-face-model/get_face_model/eve/metrics/frame_consistency/pos_hcs_consistency.py
"""
import json
import os
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

log = get_logger('preprocess.specific_face_model.eve.pos_hcs_consistency')

LM_ROOT = Path('/media/yanglinxuan/zyx/EVE_dataset/eve_dataset/landmarks')
RAW_ROOT = Path('/media/yanglinxuan/zyx/EVE_dataset/eve_dataset')
FM_DIR = Path('/media/yanglinxuan/sfm/eve_specific_face_model/face_models')
N_FRAMES = 120
DUMMY = np.zeros((32, 32, 3), np.uint8)
REF_CAM = 0                       # basler，固定参考
CACHE = HERE / 'pos_hcs_cache.npz'


def sample():
    """逐对采样：返回带相机/被试标签的原始值序列"""
    GEN6 = np.loadtxt(
        PROJECT_ROOT / 'preprocess/zhang2015-insightface/face_model_xgaze.txt'
    )[[20, 23, 26, 29, 15, 19], :]

    subjects = [(sp, p.stem) for sp in ('train', 'test')
                for p in sorted(LM_ROOT.joinpath(sp).glob('*.h5'))]
    P_g, H_g, H_t = [], [], []
    cam_g, cam_t, subj_g, subj_t, n_frames = [], [], [], [], []
    for sp, subj in tqdm(subjects, desc='subjects', unit='subj'):
        mpath = FM_DIR / subj / 'true6.txt'
        if not mpath.is_file():
            continue
        si = len(n_frames)
        true6 = np.loadtxt(mpath)
        with h5py.File(LM_ROOT / sp / f'{subj}.h5', 'r') as f:
            fr_all = f['frame_index'][:].ravel()
            ci_all = f['cam_index'][:].ravel()
            st_all = f['step_index'][:].ravel()
            lm_all = f['facial_landmarks_2d'][:]
            cameras = json.loads(f.attrs['cameras'])
            steps = json.loads(f.attrs['steps'])
        # 官方内参/外参（跨 step 恒定，取首个在盘 step）
        Ks, Rs, ts = {}, {}, {}
        for c, cam_name in enumerate(cameras):
            for step in steps:
                p = RAW_ROOT / subj / step / f'{cam_name}.h5'
                if p.is_file():
                    with h5py.File(p, 'r') as f:
                        Ks[c] = np.array(f['camera_matrix'], dtype=float)
                        T = np.array(f['camera_transformation'], dtype=float)
                    Rs[c] = T[:3, :3]
                    ts[c] = T[:3, 3].reshape(3, 1)
                    break
        if len(Ks) < 3:
            continue

        # 帧同步：basler(cam0) raw//2 对齐 webcam
        sync_map = {}
        for r in range(len(fr_all)):
            c, raw_f, si_ = int(ci_all[r]), int(fr_all[r]), int(st_all[r])
            sync_f = raw_f // 2 if c == 0 else raw_f
            sync_map.setdefault((sync_f, si_), []).append((c, r, raw_f, si_))
        groups = [v for v in sync_map.values()
                  if len(set(x[0] for x in v)) >= 3]
        if not groups:
            continue
        idx = np.linspace(0, len(groups) - 1,
                          min(N_FRAMES, len(groups))).astype(int)
        n_used = 0

        for gi in idx:
            rows_raw = groups[gi]
            step_name = steps[rows_raw[0][3]]
            # 逐相机官方 PoG（各自 raw_f）：屏幕 px → 相机系 3D 注视点（§4 公式）
            gp_cam = {}
            for c, r, raw_f, _ in rows_raw:
                p_h5 = RAW_ROOT / subj / step_name / f'{cameras[c]}.h5'
                if not p_h5.is_file():
                    continue
                with h5py.File(p_h5, 'r') as f:
                    if raw_f >= len(f['face_PoG_tobii/validity']) or \
                            not f['face_PoG_tobii/validity'][raw_f]:
                        continue
                    PoG = np.array(f['face_PoG_tobii/data'][raw_f])
                    mmpp = np.array(f['millimeters_per_pixel'], dtype=float)
                    T = np.array(f['camera_transformation'], dtype=float)
                gp_cam[c] = (T @ np.array(
                    [PoG[0] * mmpp[0], PoG[1] * mmpp[1], 0., 1.]))[:3].reshape(3, 1)
            if set(gp_cam) != set(range(4)):
                continue     # 组门控：四相机齐全且 PoG 全有效，缺一即弃

            # ---- gen6 臂：逐相机 PnP → 世界系 3D + HCS ----
            Xws, HCSg = {}, {}
            for c, r, raw_f, _ in rows_raw:
                if c not in Ks or c not in gp_cam:
                    continue
                try:
                    rvec, tvec = estimateHeadPose(
                        lm_all[r][core.IDX6].reshape(6, 1, 2).astype(float),
                        GEN6, Ks[c], None)
                except cv2.error:
                    continue
                hR = cv2.Rodrigues(rvec)[0]
                Xws[c] = (Rs[c].T @ (hR @ GEN6.T + tvec.reshape(3, 1)
                                      - ts[c])).T
                _, hr, gc = normalizeData_face(
                    DUMMY, GEN6, rvec, tvec, gp_cam[c], Ks[c],
                    fixed_forward=False)[:3]
                HCSg[c] = (cv2.Rodrigues(hr)[0].T @ gc).ravel()
            if REF_CAM in Xws:
                for c in sorted(Xws):
                    if c == REF_CAM:
                        continue
                    P_g.append(float(np.linalg.norm(
                        Xws[c] - Xws[REF_CAM], axis=1).mean()))
                    cos = float(np.clip(
                        HCSg[c] @ HCSg[REF_CAM]
                        / (np.linalg.norm(HCSg[c]) * np.linalg.norm(HCSg[REF_CAM])),
                        -1, 1))
                    H_g.append(float(np.degrees(np.arccos(cos))))
                    cam_g.append(c)
                    subj_g.append(si)

            # ---- true6 臂：4 台一组 DLT → Kabsch(true6) 共享头姿 → HCS ----
            rays, pv = [], []
            for c, r, _, _ in rows_raw:
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
            n_used += 1
            X = core.triangulate(np.stack(rays), np.stack(pv), n_points=6)
            R_h, t_h = core.kabsch(true6, X)
            HCSt = {}
            for c, r, raw_f, _ in rows_raw:
                if c not in Ks or c not in gp_cam:
                    continue
                rvec = cv2.Rodrigues(Rs[c] @ R_h)[0]
                tvec = Rs[c] @ t_h.reshape(3, 1) + ts[c]
                _, hr, gc = normalizeData_face(
                    DUMMY, true6, rvec, tvec, gp_cam[c], Ks[c],
                    fixed_forward=False)[:3]
                HCSt[c] = (cv2.Rodrigues(hr)[0].T @ gc).ravel()
            if REF_CAM in HCSt:
                for c in sorted(HCSt):
                    if c == REF_CAM:
                        continue
                    cos = float(np.clip(
                        HCSt[c] @ HCSt[REF_CAM]
                        / (np.linalg.norm(HCSt[c]) * np.linalg.norm(HCSt[REF_CAM])),
                        -1, 1))
                    H_t.append(float(np.degrees(np.arccos(cos))))
                    cam_t.append(c)
                    subj_t.append(si)
        n_frames.append((subj, n_used))

    np.savez(CACHE,
             pos3d_gen6=np.array(P_g), hcs_gen6=np.array(H_g),
             hcs_true6=np.array(H_t), cam_g=np.array(cam_g),
             cam_t=np.array(cam_t), subj_g=np.array(subj_g),
             subj_t=np.array(subj_t),
             subj_names=np.array([n for n, _ in n_frames]),
             n_frames=np.array([n for _, n in n_frames]))
    log.info(f'缓存写入 {CACHE.name}（{len(H_t):,} 对）')


def main():
    if CACHE.is_file() and not os.environ.get('POS_HCS_REFRESH'):
        z = np.load(CACHE, allow_pickle=False)
        log.info(f'缓存命中 {CACHE.name}（{len(z["hcs_true6"]):,} 对）')
    else:
        sample()
        z = np.load(CACHE)
    P_g, H_g, H_t = z['pos3d_gen6'], z['hcs_gen6'], z['hcs_true6']
    cam_g, cam_t = z['cam_g'], z['cam_t']
    subj_g, subj_t = z['subj_g'], z['subj_t']
    names, nf = list(z['subj_names']), z['n_frames']

    # ---- 逐被试 CSV ----
    overall = HERE / 'consistency_overall.csv'
    rows = []
    for si, (name, n) in enumerate(zip(names, nf)):
        mg, mt = subj_g == si, subj_t == si
        if not mg.any() or not mt.any():
            continue
        rows.append((name, int(n), np.median(P_g[mg]),
                     np.median(H_g[mg]), np.median(H_t[mt])))
    with open(overall, 'w') as f:
        f.write('subject,n_frames,pos3d_gen6_mm,hcs_gen6_deg,hcs_true6_deg\n')
        for row in rows:
            f.write(','.join(f'{v:.3f}' if isinstance(v, float) else str(v)
                             for v in row) + '\n')
        arr = np.array([r[2:] for r in rows])
        f.write(f'AVG,{np.mean([r[1] for r in rows]):.1f},'
                + ','.join(f'{v:.3f}' for v in arr.mean(0)) + '\n')

    # ---- 逐相机 CSV ----
    per_cam = HERE / 'consistency_per_camera.csv'
    with open(per_cam, 'w') as f:
        f.write('cam,n_pairs,pos3d_gen6_mm,hcs_gen6_deg,hcs_true6_deg\n')
        rows_c = []
        for c in sorted(set(cam_t.tolist())):
            if c == REF_CAM:
                continue
            mg, mt = cam_g == c, cam_t == c
            rows_c.append((c, int(mt.sum()), np.median(P_g[mg]),
                           np.median(H_g[mg]), np.median(H_t[mt])))
        for c, n, a, b, t in rows_c:
            f.write(f'cam{c:02d},{n},{a:.3f},{b:.3f},{t:.3f}\n')
        arr_c = np.array([r[2:] for r in rows_c])
        f.write('AVG vs cam00,' + str(rows_c[0][1]) + ','
                + ','.join(f'{v:.3f}' for v in arr_c.mean(0)) + '\n')

    # ---- hcs_true6 概率分布图（全体对汇总）----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5))
    hi = 5.0                                        # 横轴显示范围 ~5°（对数）
    lo = max(1e-3, H_t[H_t > 0].min() * 0.6)
    bins = np.logspace(np.log10(lo), np.log10(hi), 120)   # 对数分箱（减箱去毛刺）
    cnt, edges = np.histogram(H_t, bins=bins)
    dens = cnt / (len(H_t) * np.diff(np.log10(bins)))     # 密度/每 log10 单位
    centers = np.sqrt(edges[:-1] * edges[1:])
    ax.plot(centers, dens, lw=1.8, color='#3b7dd8')
    ax.fill_between(centers, dens, alpha=0.15, color='#3b7dd8')
    ax.set_xscale('log')
    stats = {q: float(np.percentile(H_t, q)) for q in (50, 90, 95, 98, 99)}
    vmax = float(H_t.max())                        # 最高误差：文字标注，不画线
    marks = [(f'p{q}={v:.3f}°', v, c, '--')
             for (q, v), c in zip(sorted(stats.items()),
                                  ('tab:green', 'tab:orange', 'tab:red',
                                   'tab:purple', 'tab:brown'))]
    marks.append((f'mean={H_t.mean():.3f}°', H_t.mean(), 'k', ':'))
    top = ax.get_ylim()[1]
    # 标签横向排布：右移 + y 向错开（按值排序后轮换三档高度，
    # 避免 0° 附近的 p50/mean/p90 互相压字）
    marks.sort(key=lambda m: m[1])
    ys = [top * 0.97, top * 0.82, top * 0.67]
    for i, (lbl, v, c, ls) in enumerate(marks):
        ax.axvline(v, color=c, ls=ls, lw=1.6)
        ax.text(v * 1.08, ys[i % 3], lbl, color=c, fontsize=9,
                ha='left', va='top', rotation=0)
    # 左上角：最大前 5 误差
    top10 = np.sort(H_t)[-10:][::-1]
    txt = 'Top-10 errors (deg):\n' + '\n'.join(
        f'{i + 1:2d}.  {v:.3f}' for i, v in enumerate(top10))
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, ha='left', va='top',
            fontsize=9, color='darkred', family='monospace',
            bbox=dict(fc='white', ec='darkred', alpha=0.85, pad=3))
    ax.set_xlim(lo, hi)
    # 横轴标实际角度值（非 10^x 科学计数）
    ticks = [v for v in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5)
             if lo <= v <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{v:g}' for v in ticks])
    ax.minorticks_off()
    ax.set_xlabel('HCS consistency vs cam00 (deg, log scale)')
    ax.set_ylabel('density')
    ax.set_title(f'EVE true6(DLT, 4-cam) HCS consistency distribution  '
                 f'(n={len(H_t):,} pairs, {len(rows)} subjects)')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png = HERE / 'hcs_true6_dist.png'
    fig.savefig(png, dpi=250)
    log.info(f'参考相机 cam00(basler) | 输出 {overall.name} / {per_cam.name} / '
             f'{png.name}（{len(rows)} 被试）| gen6: 3D '
             f'{np.median([r[2] for r in rows]):.2f} mm, '
             f'HCS {np.median([r[3] for r in rows]):.2f}° | '
             f'true6(4台一组): HCS {np.median([r[4] for r in rows]):.2f}° '
             f'(p90 {stats[90]:.2f}°, p95 {stats[95]:.2f}°)')


if __name__ == '__main__':
    main()
