"""离群组归因分析（通用版，2026-08-30）

用法（仓库根目录）:
  python .../exception/analyze_outlier.py subj=train08 si=61 sync=756
可选: cam=<c> 只看指定相机对（默认全对）

输出：诊断打印 + 本目录 analysis_<subj>_s<sync>.json + 各相机帧 PNG
检查项：PoG 跨相机一致性（px 离散）、DLT 逐相机重投影残差、HCS 逐对、
basler 偏移扫描（同步漂移假设）。
"""
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[6]
sys.path.insert(0, str(HERE.parents[3]))                    # get_face_model/
sys.path.insert(0, str(PROJECT_ROOT))

import face_model_core as core                              # noqa: E402
from utils.normalization import normalizeData_face          # noqa: E402

LM_ROOT = Path('/media/yanglinxuan/zyx/EVE_dataset/eve_dataset/landmarks')
RAW_ROOT = Path('/media/yanglinxuan/zyx/EVE_dataset/eve_dataset')
FM_DIR = Path('/media/yanglinxuan/sfm/eve_specific_face_model/face_models')
CAMS = ['basler', 'webcam_l', 'webcam_c', 'webcam_r']

args = dict(a.split('=') for a in sys.argv[1:])
SUBJ = args.get('subj')
SPLIT = args.get('split', 'train')
SI, SYNC_F = int(args['si']), int(args['sync'])
report = {}

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

pog = {}
for c, r, rf in rows:
    with h5py.File(raw / f'{CAMS[c]}.h5', 'r') as f:
        pog[c] = (np.array(f['face_PoG_tobii/data'][rf]),
                  bool(f['face_PoG_tobii/validity'][rf]),
                  np.array(f['millimeters_per_pixel'], dtype=float),
                  np.array(f['camera_matrix'], dtype=float),
                  np.array(f['camera_transformation'], dtype=float))
    print(f'cam{c} {CAMS[c]:9s} PoG_px={np.round(pog[c][0], 1)} valid={pog[c][1]}')
P = np.stack([pog[c][0] for c in sorted(pog)])
spread = float(np.max(np.linalg.norm(P - P.mean(0), axis=1)))
print(f'PoG 跨相机离散 = {spread:.1f} px（门控上限 20px）')
report['pog_spread_px'] = spread
report['pog_px'] = {str(c): pog[c][0].tolist() for c in pog}

# basler 偏移扫描
with h5py.File(raw / 'basler.h5', 'r') as f:
    bp = np.array(f['face_PoG_tobii/data'])
web_mean = P[1:].mean(0)
d = np.linalg.norm(bp[:2000] - web_mean, axis=1)
print(f'basler@{2*SYNC_F} 距 webcams 均值 {d[2*SYNC_F]:.1f}px；'
      f'邻近 ±15 帧最小 {d[max(0,2*SYNC_F-15):2*SYNC_F+16].min():.1f}px')
report['basler_near_min_px'] = float(d[max(0, 2*SYNC_F-15):2*SYNC_F+16].min())

# DLT + 重投影残差 + HCS
Ks = {c: pog[c][3] for c in pog}
Rs = {c: pog[c][4][:3, :3] for c in pog}
ts = {c: pog[c][4][:3, 3].reshape(3, 1) for c in pog}
rays, pv = [], []
for c, r, rf in rows:
    lm_n = cv2.undistortPoints(
        lm_all[r][core.IDX6].astype(np.float64).reshape(-1, 1, 2),
        Ks[c], None).reshape(-1, 2)
    rays.append(lm_n)
    pv.append(np.concatenate([cv2.Rodrigues(Rs[c])[0].ravel(), ts[c].ravel()]))
X = core.triangulate(np.stack(rays), np.stack(pv), n_points=6)
print('DLT 逐相机重投影残差(px):')
resid = {}
for c, r, rf in rows:
    Xc = Rs[c] @ X.T + ts[c]
    proj = (Ks[c] @ Xc).T
    proj = proj[:, :2] / proj[:, 2:3]
    obs = lm_all[r][core.IDX6].astype(float)
    err = np.linalg.norm(proj - obs, axis=1)
    resid[f'cam{c:02d}'] = err.tolist()
    print(f'  cam{c} {CAMS[c]:9s} 各点 {np.round(err, 1)}  均值 {err.mean():.1f}')
report['reproj_px'] = resid

# leave-one-out DLT：去掉每台相机后 3 台重建的 6 点位置变化（mm）
print('Leave-one-out DLT 6 点位置变化 (mm):')
loo = {}
for c_drop in sorted(pog):
    rays2 = [lm_n for (c, r, rf), lm_n in zip(rows, rays) if c != c_drop]
    pv2 = [q for (c, r, rf), q in zip(rows, pv) if c != c_drop]
    X2 = core.triangulate(np.stack(rays2), np.stack(pv2), n_points=6)
    loo[f'drop_cam{c_drop:02d}'] = float(np.linalg.norm(X2 - X, axis=1).mean())
    print(f'  去 cam{c_drop}: {loo[f"drop_cam{c_drop:02d}"]:.2f} mm')
report['loo_mm'] = loo

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
report['hcs_pairs_deg'] = {}
for c in sorted(hcs):
    if c == 0:
        continue
    cos = np.clip(hcs[c] @ hcs[0] / (np.linalg.norm(hcs[c]) * np.linalg.norm(hcs[0])), -1, 1)
    deg = float(np.degrees(np.arccos(cos)))
    report['hcs_pairs_deg'][f'cam{c:02d}_vs_cam00'] = deg
    print(f'HCS cam{c} vs cam00 = {deg:.3f}°')

for c, r, rf in rows:
    cap = cv2.VideoCapture(str(raw / f'{CAMS[c]}.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES, rf)
    ok, img = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(str(HERE / f'cam{c:02d}_{CAMS[c]}_f{rf:04d}.png'), img)

out = HERE / f'analysis_{SUBJ}_s{SYNC_F}.json'
with open(out, 'w') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f'输出 {out} + 帧PNG')
