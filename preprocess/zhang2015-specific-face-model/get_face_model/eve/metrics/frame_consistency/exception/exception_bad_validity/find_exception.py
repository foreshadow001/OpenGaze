"""[已归档 2026-08-30：坏有效性组案例（缺相机/PoG 无效），现协议已门控丢弃]
定位并搬运 hcs_true6 最大离群对（22.3°, train24 cam01 vs cam00）的原始数据

重放 pos_hcs_consistency.py 的确定性采样（无随机源，逐被试/逐组顺序固定），
在 train24 中找到该对所在的同步组，然后把原始材料搬到本目录：
- 4 相机该时刻的 mp4 帧 → cam{c}_{name}_f{raw:04d}.png
- 逐相机 PoG/validity/mmpp/camera_transformation、landmarks 行 → exception_data.npz
- train24 的 true6 / true6_canonical 模型副本
- summary.json（组信息 + 各相机 HCS + 逐对误差）
用法（仓库根目录）: python .../exception/find_exception.py
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
from utils.normalization import estimateHeadPose, normalizeData_face  # noqa: E402

log_w = print

LM_ROOT = Path('/media/yanglinxuan/zyx/EVE_dataset/eve_dataset/landmarks')
RAW_ROOT = Path('/media/yanglinxuan/zyx/EVE_dataset/eve_dataset')
FM_DIR = Path('/media/yanglinxuan/sfm/eve_specific_face_model/face_models')
N_FRAMES = 120
DUMMY = np.zeros((32, 32, 3), np.uint8)
REF_CAM = 0
SUBJ, SPLIT = 'train24', 'train'
TARGET = 22.298              # pos_hcs_cache.npz 中 hcs_true6 的最大值


def main():
    true6 = np.loadtxt(FM_DIR / SUBJ / 'true6.txt')
    with h5py.File(LM_ROOT / SPLIT / f'{SUBJ}.h5', 'r') as f:
        fr_all = f['frame_index'][:].ravel()
        ci_all = f['cam_index'][:].ravel()
        st_all = f['step_index'][:].ravel()
        lm_all = f['facial_landmarks_2d'][:]
        cameras = json.loads(f.attrs['cameras'])
        steps = json.loads(f.attrs['steps'])
    Ks, Rs, ts = {}, {}, {}
    for c, cam_name in enumerate(cameras):
        for step in steps:
            p = RAW_ROOT / SUBJ / step / f'{cam_name}.h5'
            if p.is_file():
                with h5py.File(p, 'r') as f:
                    Ks[c] = np.array(f['camera_matrix'], dtype=float)
                    T = np.array(f['camera_transformation'], dtype=float)
                Rs[c] = T[:3, :3]
                ts[c] = T[:3, 3].reshape(3, 1)
                break

    sync_map = {}
    for r in range(len(fr_all)):
        c, raw_f, si = int(ci_all[r]), int(fr_all[r]), int(st_all[r])
        sync_f = raw_f // 2 if c == 0 else raw_f
        sync_map.setdefault((sync_f, si), []).append((c, r, raw_f, si))
    groups = [v for v in sync_map.values() if len(set(x[0] for x in v)) >= 3]
    idx = np.linspace(0, len(groups) - 1, min(N_FRAMES, len(groups))).astype(int)

    hit = None
    for gi in idx:
        rows_raw = groups[gi]
        step_name = steps[rows_raw[0][3]]
        gp_cam, pog_px, mmpps = {}, {}, {}
        for c, r, raw_f, _ in rows_raw:
            p_h5 = RAW_ROOT / SUBJ / step_name / f'{cameras[c]}.h5'
            if not p_h5.is_file():
                continue
            with h5py.File(p_h5, 'r') as f:
                if raw_f >= len(f['face_PoG_tobii/validity']) or \
                        not f['face_PoG_tobii/validity'][raw_f]:
                    continue
                PoG = np.array(f['face_PoG_tobii/data'][raw_f])
                mmpp = np.array(f['millimeters_per_pixel'], dtype=float)
                T = np.array(f['camera_transformation'], dtype=float)
            pog_px[c], mmpps[c] = PoG, mmpp
            gp_cam[c] = (T @ np.array([PoG[0] * mmpp[0], PoG[1] * mmpp[1],
                                       0., 1.]))[:3].reshape(3, 1)
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
        if REF_CAM not in HCSt:
            continue
        for c in sorted(HCSt):
            if c == REF_CAM:
                continue
            cos = float(np.clip(
                HCSt[c] @ HCSt[REF_CAM]
                / (np.linalg.norm(HCSt[c]) * np.linalg.norm(HCSt[REF_CAM])), -1, 1))
            d = float(np.degrees(np.arccos(cos)))
            if abs(d - TARGET) < 0.01:
                hit = (gi, rows_raw, step_name, HCSt, gp_cam, pog_px, mmpps,
                       {k: v for k, v in sync_map.items() if v is rows_raw})
                log_w(f'命中: group#{gi} step={step_name} cam{c} 误差 {d:.3f}°')
                break
        if hit:
            break
    if not hit:
        log_w('未找到目标对（TARGET 需核对）')
        return
    gi, rows_raw, step_name, HCSt, gp_cam, pog_px, mmpps, key_info = hit
    sync_f, si = list(key_info.keys())[0]

    # ---- 搬运原始数据 ----
    summary = {'subject': SUBJ, 'split': SPLIT, 'step': step_name,
               'sync_frame': int(sync_f), 'step_index': int(si),
               'target_error_deg': TARGET, 'cameras': cameras,
               'rows': [(int(c), int(r), int(raw_f)) for c, r, raw_f, _ in rows_raw],
               'hcs': {str(c): HCSt[c].tolist() for c in HCSt},
               'pairs_deg': {}}
    for c in HCSt:
        if c == REF_CAM:
            continue
        cos = float(np.clip(HCSt[c] @ HCSt[REF_CAM]
                            / (np.linalg.norm(HCSt[c]) * np.linalg.norm(HCSt[REF_CAM])), -1, 1))
        summary['pairs_deg'][f'cam{c:02d}_vs_cam00'] = float(np.degrees(np.arccos(cos)))
    data = {'lm106': {}, 'pog_px': {}, 'mmpp': {}, 'K': {}, 'T': {},
            'gp_cam': {}, 'hcs': {}}
    for c, r, raw_f, _ in rows_raw:
        cam_name = cameras[c]
        cap = cv2.VideoCapture(str(RAW_ROOT / SUBJ / step_name / f'{cam_name}.mp4'))
        cap.set(cv2.CAP_PROP_POS_FRAMES, raw_f)
        ok, img = cap.read()
        cap.release()
        if ok:
            cv2.imwrite(str(HERE / f'cam{c:02d}_{cam_name}_f{raw_f:04d}.png'), img)
        data['lm106'][f'cam{c:02d}'] = lm_all[r]
        if c in pog_px:
            data['pog_px'][f'cam{c:02d}'] = pog_px[c]
            data['mmpp'][f'cam{c:02d}'] = mmpps[c]
            data['gp_cam'][f'cam{c:02d}'] = gp_cam[c].ravel()
        if c in Ks:
            data['K'][f'cam{c:02d}'] = Ks[c]
            T4 = np.eye(4)
            T4[:3, :3] = Rs[c]
            T4[:3, 3:] = ts[c]
            data['T'][f'cam{c:02d}'] = T4
        if c in HCSt:
            data['hcs'][f'cam{c:02d}'] = HCSt[c]
    np.savez(HERE / 'exception_data.npz',
             **{f'{k}_{ck}': v for k, d in data.items() for ck, v in d.items()})
    np.savetxt(HERE / 'true6.txt', true6, fmt='%.6f')
    np.savetxt(HERE / 'true6_canonical.txt',
               np.loadtxt(FM_DIR / SUBJ / 'true6_canonical.txt'), fmt='%.6f')
    with open(HERE / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log_w(f'搬运完成 → {HERE}（帧 png / exception_data.npz / 模型 / summary.json）')


if __name__ == '__main__':
    main()
