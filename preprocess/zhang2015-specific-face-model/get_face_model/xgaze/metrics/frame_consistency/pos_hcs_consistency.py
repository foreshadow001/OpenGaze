"""3D 位置与头姿(HCS gaze) 跨相机一致性——gen6 vs true6 双臂对比（2026-08-30）

协议与 frame_consistency.py 同（每人固定前 10 台相机 cam00..09，只用官方外参，
每相机独立 6 点 PnP）。两臂仅换 PnP/归一化所用模型：

  gen6   通用模型 face_model_xgaze.txt 6 点子集（v1 部署形态）
  true6  逐被试真实模型 true6.txt（严格三角化，gen6 对齐系）

每臂各测两项：
1. 3D 位置跨相机一致性 (mm)：
   gen6 臂——各相机 PnP 解出的 6 点经官方外参转世界系，同帧同名点跨相机
   两两欧氏距离 → 中位；
   true6 臂——偶/奇两组（各 5 台）独立 DLT 三角化，两组 6 点逐点距离均值；
2. 头姿一致性 (deg)：用 gaze HCS 实现（参考 sample_all_datasets.py 的 camd；
   HCS 与归一化旋转严格无关）——
   gen6 臂——各相机用自身 PnP 头姿独立归一化，同帧跨相机两两夹角；
   true6 臂（5+5 互验）——各组 DLT 3D 点 Kabsch 到 true6 得组头姿，组内相机
   各自合成 rvec/tvec 后归一化，取跨组相机对的两两夹角；
   true6_full（10 台一组）——v2 部署形态：全部 10 台一次 DLT 共享头姿，
   各相机仅注视标注独立 → 跨相机两两夹角（纯标注+外参一致性，预期 ~0.01°）。

输出（本目录，无 md）:
  consistency_overall.csv     逐被试中位（双臂列并排）
  consistency_per_camera.csv  逐相机 vs 其余相机中位（双臂列并排）
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
CAMS = list(range(10))                                       # 协议：前 10 台
N_FRAMES = 120                                               # 每人采样帧数（协议同旧实验）
DUMMY = np.zeros((32, 32, 3), np.uint8)
ARMS = ('gen6', 'true6')
REF_CAM = 0                       # 固定参考相机：所有"跨相机"指标 = vs cam00


def main():
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

    sids = sorted(p.stem for p in Path(LM_DIR).glob('subject*.h5'))
    subj_rows = []
    cam_acc = {arm: {c: ([], []) for c in CAMS} for arm in ARMS}   # (pos, hcs)
    cam_full = {c: [] for c in CAMS}                              # true6_full HCS
    subj_full = []                                                # 逐被试中位
    for sid in tqdm(sids, desc='subjects', unit='subj'):
        true6 = np.loadtxt(FM_DIR / sid / 'true6.txt')
        models = {'gen6': GEN6, 'true6': true6}
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

        # 每臂独立累积：pos / hcs 的逐对值
        vals = {arm: ([], []) for arm in ARMS}
        vals_full = []                 # true6_full（10 台一组）HCS 逐对夹角
        for fidx in (frames[i] for i in idx):
            rows_f = by_frame[fidx]
            # ---- gen6 臂：逐相机独立 PnP ----
            model = models['gen6']
            Xws, HCSs = {}, {}
            for c, r in rows_f:
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
                                       - TR[c])).T              # (6,3)
                _, hr, gc = normalizeData_face(
                    DUMMY, model, rvec, tvec, gp, KS[c],
                    fixed_forward=False)[:3]
                HCSs[c] = (cv2.Rodrigues(hr)[0].T @ gc).ravel()
            if len(Xws) >= 4 and REF_CAM in Xws:
                for c in sorted(Xws):
                    if c == REF_CAM:
                        continue
                    d_pos = float(np.linalg.norm(
                        Xws[c] - Xws[REF_CAM], axis=1).mean())
                    cos = float(np.clip(
                        HCSs[c] @ HCSs[REF_CAM]
                        / (np.linalg.norm(HCSs[c])
                           * np.linalg.norm(HCSs[REF_CAM])), -1, 1))
                    d_hcs = float(np.degrees(np.arccos(cos)))
                    vals['gen6'][0].append(d_pos)
                    vals['gen6'][1].append(d_hcs)
                    cam_acc['gen6'][c][0].append(d_pos)
                    cam_acc['gen6'][c][1].append(d_hcs)

            # ---- true6 臂：偶/奇两组独立 DLT → Kabsch(true6) 解头姿 ----
            model = models['true6']
            Xg, HCSg = {}, {}                    # 组 -> (6,3) 世界点 / {cam: HCS}
            for g, cams_g in (('A', [0, 2, 4, 6, 8]), ('B', [1, 3, 5, 7, 9])):
                rays, pv = [], []
                for c, r in rows_f:
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
                for c, r in rows_f:               # 组内相机各自合成 + 独立标注
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
                    HCSg[c] = (cv2.Rodrigues(hr)[0].T @ gc).ravel()
            # ---- true6_full：10 台一组 DLT 共享头姿（v2 部署形态）----
            model = models['true6']
            rays, pv = [], []
            for c, r in rows_f:
                lm_n = cv2.undistortPoints(
                    lm_all[r][core.IDX6].astype(np.float64).reshape(-1, 1, 2),
                    KS[c], DIST[c]).reshape(-1, 2)
                rays.append(lm_n)
                pv.append(np.concatenate(
                    [cv2.Rodrigues(ROT[c])[0].ravel(), TR[c].ravel()]))
            if len(rays) >= 3:
                X = core.triangulate(np.stack(rays), np.stack(pv), n_points=6)
                R_h, t_h = core.kabsch(model, X)
                Hf = {}
                for c, r in rows_f:
                    gp = ann.get((f'frame{fidx:04d}', f'cam{c:02d}.JPG'))
                    if gp is None:
                        continue
                    rvec = cv2.Rodrigues(ROT[c] @ R_h)[0]
                    tvec = ROT[c] @ t_h.reshape(3, 1) + TR[c]
                    _, hr, gc = normalizeData_face(
                        DUMMY, model, rvec, tvec, gp, KS[c],
                        fixed_forward=False)[:3]
                    Hf[c] = (cv2.Rodrigues(hr)[0].T @ gc).ravel()
                if REF_CAM in Hf:
                    for c in sorted(Hf):
                        if c == REF_CAM:
                            continue
                        cos = float(np.clip(
                            Hf[c] @ Hf[REF_CAM]
                            / (np.linalg.norm(Hf[c])
                               * np.linalg.norm(Hf[REF_CAM])), -1, 1))
                        d = float(np.degrees(np.arccos(cos)))
                        vals_full.append(d)
                        cam_full[c].append(d)

            if 'A' in Xg and 'B' in Xg:
                d_pos = float(np.linalg.norm(Xg['A'] - Xg['B'], axis=1).mean())
                vals['true6'][0].append(d_pos)
                if REF_CAM in HCSg:               # 一致性 = vs cam00（固定参考）
                    for c, h in HCSg.items():
                        if c == REF_CAM:
                            continue
                        cos = float(np.clip(
                            h @ HCSg[REF_CAM]
                            / (np.linalg.norm(h)
                               * np.linalg.norm(HCSg[REF_CAM])), -1, 1))
                        d_hcs = float(np.degrees(np.arccos(cos)))
                        vals['true6'][1].append(d_hcs)
                        cam_acc['true6'][c][0].append(d_pos)
                        cam_acc['true6'][c][1].append(d_hcs)
        if vals['gen6'][0] and vals['true6'][0]:
            subj_rows.append(
                (sid, len(idx))
                + tuple(np.median(vals[arm][k]) for arm in ARMS for k in (0, 1)))
            subj_full.append((sid, np.median(vals_full) if vals_full else np.nan))

    full_med = {sid: v for sid, v in subj_full}
    overall = HERE / 'consistency_overall.csv'
    with open(overall, 'w') as f:
        f.write('subject,n_frames,pos3d_gen6_mm,hcs_gen6_deg,'
                'pos3d_true6_mm,hcs_true6_deg,hcs_true6_full_deg\n')
        for row in subj_rows:
            f.write(f'{row[0]},{row[1]},'
                    + ','.join(f'{v:.3f}' for v in row[2:]) + ','
                    + f'{full_med[row[0]]:.3f}' + '\n')
        arr = np.array([list(r[2:]) + [full_med[r[0]]] for r in subj_rows])
        f.write(f'AVG,{np.mean([r[1] for r in subj_rows]):.1f},'
                + ','.join(f'{v:.3f}' for v in arr.mean(0)) + '\n')
    per_cam = HERE / 'consistency_per_camera.csv'
    with open(per_cam, 'w') as f:
        f.write('cam,n_pairs,pos3d_gen6_mm,hcs_gen6_deg,'
                'pos3d_true6_mm,hcs_true6_deg,hcs_true6_full_deg\n')
        rows_c = []
        for c in CAMS:
            if c == REF_CAM:                     # 参考相机自比无意义，不输出
                continue
            g, t = cam_acc['gen6'][c], cam_acc['true6'][c]
            if g[0]:
                rows_c.append((c, len(g[0]), np.median(g[0]), np.median(g[1]),
                               np.median(t[0]), np.median(t[1]),
                               np.median(cam_full[c]) if cam_full[c] else np.nan))
        for c, n, a, b, d, e, fc in rows_c:
            f.write(f'cam{c:02d},{n},{a:.3f},{b:.3f},{d:.3f},{e:.3f},{fc:.3f}\n')
        arr_c = np.array([r[2:] for r in rows_c])
        f.write('AVG vs cam00,' + str(rows_c[0][1]) + ','
                + ','.join(f'{v:.3f}' for v in arr_c.mean(0)) + '\n')
    med = {arm: (np.median([r[2 + 2 * i] for r in subj_rows]),
                 np.median([r[3 + 2 * i] for r in subj_rows]))
           for i, arm in enumerate(ARMS)}
    med_full = np.median([v for _, v in subj_full])
    log.info(f'参考相机 cam00（全部指标 = vs cam00）| '
             f'输出 {overall.name} / {per_cam.name}（{len(subj_rows)} 被试）| '
             f'gen6:  3D {med["gen6"][0]:.2f} mm, HCS {med["gen6"][1]:.2f}° | '
             f'true6(5+5): 3D {med["true6"][0]:.2f} mm, HCS {med["true6"][1]:.2f}° | '
             f'true6_full(10 台): HCS {med_full:.2f}°')


if __name__ == '__main__':
    main()
