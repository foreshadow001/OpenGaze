"""[已归档 2026-08-30：PoG 跨相机离散（标注时间错位）案例，现协议已门控丢弃]
离群组归因分析：train22 / step95 / sync_f=492（HCS 15.7°/13.9°）

假设检验：
  H1 同步漂移：basler 帧号 //2 在该时刻不成立（掉帧→basler@984 与 webcam@492
     非同一物理时刻）→ basler 的 PoG 屏幕点与 webcams 不同
  H2 特征点离群：某相机 6 点关键点坏 → DLT 被拉偏 / PnP 错
  H3 标注本身离群：PoG 数值跨相机不一致（非同流分发）
输出：诊断打印 + 本目录 4 帧PNG + analysis.json
"""
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[7]
sys.path.insert(0, str(HERE.parents[4]))                    # get_face_model/
sys.path.insert(0, str(PROJECT_ROOT))

import face_model_core as core                              # noqa: E402
from utils.normalization import normalizeData_face          # noqa: E402

LM_ROOT = Path('/media/yanglinxuan/zyx/EVE_dataset/eve_dataset/landmarks')
RAW_ROOT = Path('/media/yanglinxuan/zyx/EVE_dataset/eve_dataset')
FM_DIR = Path('/media/yanglinxuan/sfm/eve_specific_face_model/face_models')
SUBJ, SPLIT, SI, SYNC_F = 'train22', 'train', 95, 492
CAMS = ['basler', 'webcam_l', 'webcam_c', 'webcam_r']

report = {}

# ---- 组信息 ----
with h5py.File(LM_ROOT / SPLIT / f'{SUBJ}.h5', 'r') as f:
    fr = f['frame_index'][:].ravel(); ci = f['cam_index'][:].ravel()
    st = f['step_index'][:].ravel(); lm_all = f['facial_landmarks_2d'][:]
    steps = json.loads(f.attrs['steps'])
step_name = steps[SI]
print(f'组: {SUBJ} {step_name} sync_f={SYNC_F}')
rows = [(int(ci[r]), r, int(fr[r])) for r in range(len(fr))
        if st[r] == SI and (int(fr[r]) // 2 if int(ci[r]) == 0 else int(fr[r])) == SYNC_F]
print('组内相机/原始帧:', [(f'cam{c}', rf) for c, _, rf in rows])
report['rows'] = [(c, rf) for c, _, rf in rows]

raw = RAW_ROOT / SUBJ / step_name
true6 = np.loadtxt(FM_DIR / SUBJ / 'true6.txt')

# ---- H1: PoG 跨相机一致性 + basler 偏移扫描 ----
pog = {}
for c, r, rf in rows:
    with h5py.File(raw / f'{CAMS[c]}.h5', 'r') as f:
        pog[c] = np.array(f['face_PoG_tobii/data'][rf]), \
            bool(f['face_PoG_tobii/validity'][rf]), \
            np.array(f['millimeters_per_pixel'], dtype=float), \
            np.array(f['camera_matrix'], dtype=float), \
            np.array(f['camera_transformation'], dtype=float)
    print(f'cam{c} {CAMS[c]:9s} PoG_px={np.round(pog[c][0], 1)} valid={pog[c][1]}')
web_pog = np.mean([pog[c][0] for c in (1, 2, 3)], axis=0)
print(f'webcams PoG 均值 = {np.round(web_pog, 1)}  '
      f'basler 偏差 = {np.linalg.norm(pog[0][0] - web_pog):.1f} px')
report['pog_px'] = {str(c): pog[c][0].tolist() for c in pog}

# basler 偏移扫描：哪个 basler 帧号的 PoG 与 webcams 一致？
with h5py.File(raw / 'basler.h5', 'r') as f:
    bp = np.array(f['face_PoG_tobii/data'])
d = np.linalg.norm(bp[:2000] - web_pog, axis=1)
best = int(np.argmin(d))
print(f'H1 同步漂移: basler@{2*SYNC_F} 偏差 {d[2*SYNC_F]:.1f}px；'
      f'最小偏差帧 @{best}（{d[best]:.1f}px, 偏移 {best - 2*SYNC_F:+d} 帧 '
      f'= {(best - 2*SYNC_F)/30*1000:+.0f}ms）')
report['h1_basler_best_frame'] = [best, float(d[best]), best - 2 * SYNC_F]

# ---- H2: DLT + 逐相机重投影残差 + HCS ----
Ks, Rs, ts = {}, {}, {}
for c in range(4):
    Ks[c], Rs[c], ts[c] = pog[c][3], pog[c][4][:3, :3], pog[c][4][:3, 3].reshape(3, 1)
rays, pv = [], []
for c, r, rf in rows:
    lm_n = cv2.undistortPoints(
        lm_all[r][core.IDX6].astype(np.float64).reshape(-1, 1, 2),
        Ks[c], None).reshape(-1, 2)
    rays.append(lm_n)
    pv.append(np.concatenate([cv2.Rodrigues(Rs[c])[0].ravel(), ts[c].ravel()]))
X = core.triangulate(np.stack(rays), np.stack(pv), n_points=6)
print('DLT 6 点两两相机重投影残差(px):')
resid = {}
for (c, r, rf), in zip(rows):
    Xc = Rs[c] @ X.T + ts[c]
    proj = (Ks[c] @ Xc).T
    proj = proj[:, :2] / proj[:, 2:3]
    obs = lm_all[r][core.IDX6].astype(float)
    err = np.linalg.norm(proj - obs, axis=1)
    resid[f'cam{c:02d}'] = err.tolist()
    print(f'  cam{c} {CAMS[c]:9s} 各点 {np.round(err, 1)}  均值 {err.mean():.1f}')
report['reproj_px'] = resid

R_h, t_h = core.kabsch(true6, X)
DUMMY = np.zeros((32, 32, 3), np.uint8)
hcs = {}
for c, r, rf in rows:
    gp = (pog[c][4] @ np.array([pog[c][0][0] * pog[c][2][0],
                                pog[c][0][1] * pog[c][2][1], 0., 1.]))[:3].reshape(3, 1)
    rvec = cv2.Rodrigues(Rs[c] @ R_h)[0]
    tvec = Rs[c] @ t_h.reshape(3, 1) + ts[c]
    _, hr, gc = normalizeData_face(DUMMY, true6, rvec, tvec, gp, Ks[c],
                                   fixed_forward=False)[:3]
    hcs[c] = (cv2.Rodrigues(hr)[0].T @ gc).ravel()
for c in hcs:
    if c == 0:
        continue
    cos = np.clip(hcs[c] @ hcs[0] / (np.linalg.norm(hcs[c]) * np.linalg.norm(hcs[0])), -1, 1)
    print(f'HCS cam{c} vs cam00 = {np.degrees(np.arccos(cos)):.3f}°')

# ---- 帧导出 ----
for c, r, rf in rows:
    cap = cv2.VideoCapture(str(raw / f'{CAMS[c]}.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES, rf)
    ok, img = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(str(HERE / f'cam{c:02d}_{CAMS[c]}_f{rf:04d}.png'), img)

with open(HERE / 'analysis.json', 'w') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f'输出 {HERE / "analysis.json"} + 4 帧PNG')
