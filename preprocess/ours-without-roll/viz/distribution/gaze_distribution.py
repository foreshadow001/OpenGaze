"""四数据集 gaze/头姿分布热力图（ours-without-roll v3，2026-09-02）

与 v2 版 [../../../zhang2015-specific-face-model/viz/distribution/gaze_distribution.py]
缓存与排版一致（逐数据集 gd_cache_<ds>.npz、GAZE_DIST_REFRESH 选择刷新、
turbo 热力图、±120°/0.5° 格、密度归一、CSV 统计），差异：

1. **几何全部取自 v2 产物 h5**（sfm/*_specific_224），无 DLT / 无 PnP /
   无图像 I/O——纯 numpy：hR,t = Kabsch(model, face_landmarks_3d)、
   gp 方向 = face_mat_norm^T·unit(face_gaze)、归一化 fixed_forward=True。
2. **行语义（5 行）**：
   - CCS gaze：v3 归一化相机系（roll 修正、头姿保留）——与 v2 版 CCS 分布不同；
   - Head pose (norm) / (raw)：v3 下欧拉 (α,β) 归一化前后严格不变，两行同值
     （分布级验证 v3 性质）。**xgaze 例外：两行均用世界系 elev/azim**
     （穹顶侧视相机的 per-camera 欧拉在眼线∥光轴处退化，且 v3 下 norm≡raw，
     世界系（=cam00 系）无极点问题——见 v2 版 exception/pitch60_tail）；
   - Head pose (world)：**新增行**（raw 头姿的世界系 elev/azim，协议
     face_head_elev_azim 口径，elev 加 −30° 零位）——xgaze=穹顶系、EVE=basler
     相机系（cam 0）、GC/MPII=相机本身；多相机同帧各相机反解同值；
   - HCS gaze：hRᵀ·gc 与归一化无关，数值与 v2 版一致（跨版对照）。

输出: 本目录 gaze_distribution_noroll.png + gaze_distribution_stats.csv
用法（仓库根目录）: python preprocess/ours-without-roll/viz/distribution/gaze_distribution.py
  GAZE_DIST_REFRESH=xgaze 只刷指定数据集；=1|all 全刷；缺省命中缓存
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

_PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_PROJECT / 'preprocess/zhang2015-specific-face-model/get_face_model'))

import face_model_core as core
from utils.logger import get_logger
from utils.normalization import (HEAD_PITCH_OFFSET, normalizeData_face,
                                 vector_to_angles)

log = get_logger('preprocess.ours_without_roll.gaze_dist')

_here = Path(__file__).resolve().parent
DUMMY = np.zeros((32, 32, 3), np.uint8)   # 纯几何，无图像 I/O
LIM, BINS = 120, 480        # ±120°，0.5° 一格（超界丢弃不堆积）
NOISE_MIN_SAMPLES = 3       # 可见门限：池化格内 ≥3 个样本
GP_SCALE = 300.0            # gp = face_center + dir·300（normalizeData 只用方向）
TAGS = ('CCS', 'HEAD', 'HEAD_RAW', 'HEAD_WORLD', 'HCS')


def unit_from_angles(theta, phi):
    """vector_to_angles 的逆：单位方向向量"""
    return np.array([-np.cos(theta) * np.sin(phi), -np.sin(theta),
                     -np.cos(theta) * np.cos(phi)])


def face_center_of(X_cam):
    """normalizeData_face 同式的 6 点加权中心"""
    two_eye = X_cam[0:4].mean(axis=0)
    nose = X_cam[4:6].mean(axis=0)
    return (two_eye + nose) / 2.0


def v3_angles(model, X_cam, matn, gaze, K, ext):
    """(ccs, euler_head, world, hcs) 度数——v3 链路（fixed_forward=True）"""
    hR, t = core.kabsch(model, X_cam)
    rvec = cv2.Rodrigues(hR)[0]
    tvec = t.reshape(3, 1)
    gp_dir = matn.T @ unit_from_angles(gaze[0], gaze[1])
    gp = (face_center_of(X_cam) + gp_dir * GP_SCALE).reshape(3, 1)
    _, hr_norm, gc = normalizeData_face(
        DUMMY, model, rvec, tvec, gp, K, fixed_forward=True)[:3]
    hR_n = cv2.Rodrigues(hr_norm)[0]
    e = Rotation.from_matrix(hR_n).as_euler('xyz', degrees=True)   # (α,β,γ)
    R_hw = ext @ hR if ext is not None else hR          # 世界←相机（None=相机即世界）
    t_w, p_w = vector_to_angles(R_hw @ np.array([0., 0., -1.]))
    return (np.degrees(vector_to_angles(gc.ravel())),
            (e[0], e[1]),                                # 欧拉 (α,β)，norm≡raw
            (np.degrees(t_w) + HEAD_PITCH_OFFSET, np.degrees(p_w)),
            np.degrees(vector_to_angles((hR_n.T @ gc).ravel())))


# ------------------------------------------------------------ XGaze
def sample_xgaze(rng, k_frame=300):
    V2 = Path('/media/yanglinxuan/sfm/xgaze_specific_224')
    CAL = Path('/media/yanglinxuan/Expansion/xgaze_raw/calibration/cam_calibration')

    KS, ROT = {}, {}
    for c in range(18):
        fs = cv2.FileStorage(str(CAL / f'cam{c:02d}.xml'), cv2.FILE_STORAGE_READ)
        KS[c] = fs.getNode('Camera_Matrix').mat()
        ROT[c] = fs.getNode('cam_rotation').mat()
        fs.release()

    out = []
    for sid in tqdm(sorted(p.stem for p in V2.glob('subject*.h5')),
                    desc='xgaze', unit='subj'):
        with h5py.File(V2 / f'{sid}.h5', 'r') as f:
            fr = f['frame_index'][:].ravel()
            ci = f['cam_index'][:].ravel()
            gaze = f['face_gaze'][:]
            lm3d = f['face_landmarks_3d'][:]
            matn = f['face_mat_norm'][:]
            model = np.array(f.attrs['face_model'])
        by_frame = {}
        for r in range(len(fr)):
            by_frame.setdefault(int(fr[r]), []).append(r)
        frames = sorted(by_frame)
        picked = rng.choice(frames, size=min(k_frame, len(frames)), replace=False)
        for fidx in picked:
            for r in by_frame[fidx]:                     # 全部相机入样（v2 口径）
                c = int(ci[r])
                out.append((model, lm3d[r], matn[r], gaze[r], KS[c], ROT[c].T))
    return out


# ------------------------------------------------------------ EVE
def sample_eve(rng, k_frame=110):
    V2 = Path('/media/yanglinxuan/sfm/eve_specific_224')
    RAW = Path('/media/yanglinxuan/zyx/EVE_dataset/eve_dataset')

    subs = [(sp, p.stem) for sp in ('train', 'test')
            for p in sorted(V2.joinpath(sp).glob('*.h5'))]
    cal_cache = {}                                       # (subj, step) → {cam: (K, ext)}
    out = []
    for sp, subj in tqdm(subs, desc='eve', unit='subj'):
        with h5py.File(V2 / sp / f'{subj}.h5', 'r') as f:
            fr = f['frame_index'][:].ravel()
            ci = f['cam_index'][:].ravel()
            st = f['step_index'][:].ravel()
            gaze = f['face_gaze'][:]
            lm3d = f['face_landmarks_3d'][:]
            matn = f['face_mat_norm'][:]
            model = np.array(f.attrs['face_model'])
            cameras = json.loads(f.attrs['cameras'])
            steps = json.loads(f.attrs['steps'])
        by_sync = {}                                     # basler(c0) 帧号÷2 对齐 webcam
        for r in range(len(fr)):
            c, raw_f, si = int(ci[r]), int(fr[r]), int(st[r])
            by_sync.setdefault((raw_f // 2 if c == 0 else raw_f, si), []).append(r)
        groups = sorted(by_sync.values(), key=len, reverse=True)
        groups = [g for g in groups if len(g) >= 3]
        picked = rng.choice(len(groups), size=min(k_frame, len(groups)),
                            replace=False)
        for gi in picked:
            rows = groups[gi]
            step = steps[int(st[rows[0]])]
            key = (subj, step)
            if key not in cal_cache:                     # 该 step 全部相机标定
                d = RAW / subj / step
                cal = {}
                try:
                    with h5py.File(d / 'basler.h5', 'r') as fb:
                        R_b = np.array(fb['camera_transformation'],
                                       dtype=float)[:3, :3]
                    for cam_name in cameras:
                        with h5py.File(d / f'{cam_name}.h5', 'r') as fc:
                            K = np.array(fc['camera_matrix'], dtype=float)
                            R_c = np.array(fc['camera_transformation'],
                                           dtype=float)[:3, :3]
                        cal[cam_name] = (K, R_b @ R_c.T)  # 世界系 = basler 系
                except (OSError, KeyError):
                    cal = None
                cal_cache[key] = cal
            cal = cal_cache[key]
            if cal is None:
                continue
            for r in rows:
                K, ext = cal[cameras[int(ci[r])]]
                out.append((model, lm3d[r], matn[r], gaze[r], K, ext))
    return out


# ------------------------------------------------------------ GazeCapture
def sample_gc(rng, k_frame=500, n_sess=600):
    V2 = Path('/media/yanglinxuan/sfm/gazecapture_specific_224')
    RAW = Path('/media/yanglinxuan/zyx/GazeCapture')
    CAL = Path('/media/yanglinxuan/zyx/GazeCapture/calibration')

    sess = [(sp, p.stem) for sp in ('train', 'test')
            for p in sorted(V2.joinpath(sp).glob('*.h5'))]
    k_cache = {}                                         # (device, w, h) → K
    out = []
    for sp, s in tqdm(sess, desc='gazecapture', unit='sess'):
        try:
            device = json.load(open(RAW / s / 'info.json'))['DeviceName'].lower() \
                .replace(' ', '-')
        except Exception:
            continue
        with h5py.File(V2 / sp / f'{s}.h5', 'r') as f:
            ori = f['orientation'][:].ravel()
            gaze = f['face_gaze'][:]
            lm3d = f['face_landmarks_3d'][:]
            matn = f['face_mat_norm'][:]
            model = np.array(f.attrs['face_model'])
        idx = rng.choice(len(gaze), size=min(k_frame, len(gaze)), replace=False)
        for i in idx:
            o = int(ori[i])
            w, h = (480, 640) if o in (1, 2) else (640, 480)
            if (device, w, h) not in k_cache:
                cal = CAL / f'{device}_{w}x{h}.xml'
                if not cal.is_file():
                    k_cache[(device, w, h)] = None
                else:
                    fs = cv2.FileStorage(str(cal), cv2.FILE_STORAGE_READ)
                    k_cache[(device, w, h)] = fs.getNode('Camera_Matrix').mat()
                    fs.release()
            K = k_cache[(device, w, h)]
            if K is None:
                continue
            out.append((model, lm3d[i], matn[i], gaze[i], K, None))
        if len(out) >= n_sess * k_frame:
            break
    return out


# ------------------------------------------------------------ MPIIFaceGaze
def sample_mpii(rng, k_frame=4000):
    V2 = Path('/media/yanglinxuan/sfm/mpiifacegaze_specific_224')
    RAW = Path('/media/yanglinxuan/zyx/MPIIFaceGaze')
    import scipy.io as sio

    out = []
    for subj in tqdm(sorted(p.stem for p in V2.glob('*.h5')), desc='mpii',
                     unit='subj'):
        with h5py.File(V2 / f'{subj}.h5', 'r') as f:
            gaze = f['face_gaze'][:]
            lm3d = f['face_landmarks_3d'][:]
            matn = f['face_mat_norm'][:]
            model = np.array(f.attrs['face_model'])
        mat = sio.loadmat(RAW / subj / 'Calibration' / 'Camera.mat')
        K = np.array(mat['cameraMatrix'], dtype=float)
        idx = rng.choice(len(gaze), size=min(k_frame, len(gaze)), replace=False)
        for i in idx:
            out.append((model, lm3d[i], matn[i], gaze[i], K, None))
    return out


# ------------------------------------------------------------ 出图
def main():
    rng = np.random.default_rng(7)
    plan = [('xgaze', sample_xgaze), ('eve', sample_eve),
            ('gazecapture', sample_gc), ('mpiifacegaze', sample_mpii)]
    BG = '#101d4a'
    cmap = matplotlib.colormaps['turbo'].copy()
    cmap.set_under(BG)
    cmap.set_bad(BG)

    refresh = os.environ.get('GAZE_DIST_REFRESH', '')
    refresh_all = refresh.lower() in ('1', 'all')
    refresh_set = set() if refresh_all else {x for x in refresh.split(',') if x}
    hists, ns = {}, {}

    for name, fn in plan:
        cfile = _here / f'gd_cache_{name}.npz'
        if not refresh_all and name not in refresh_set and cfile.is_file():
            z = np.load(cfile)
            if all(t in z.files for t in TAGS):
                ns[name] = int(z['ns'])
                for t in TAGS:
                    hists[(name, t)] = z[t]
                log.info(f'缓存命中 {name}（n={ns[name]:,}）')
                continue
        items = fn(rng)
        res = [v3_angles(*it) for it in
               tqdm(items, desc=f'{name} 角度', unit='样本', leave=False)]
        ns[name] = len(res)
        log.info(f'{name}: {ns[name]:,} 样本（(frame,cam) 口径）')
        arrs = {'CCS': np.array([r[0] for r in res]),
                'EULER': np.array([r[1] for r in res]),
                'HEAD_WORLD': np.array([r[2] for r in res]),
                'HCS': np.array([r[3] for r in res])}
        # xgaze 头姿两行（norm/raw）均用世界系（穹顶 per-camera 欧拉退化且 v3 下同值）；
        # 其余数据集 norm/raw 用欧拉 (α,β)（v3 下严格同值），另加世界系行
        if name == 'xgaze':
            arrs['HEAD'] = arrs['HEAD_RAW'] = arrs['HEAD_WORLD']
        else:
            arrs['HEAD'] = arrs['HEAD_RAW'] = arrs['EULER']
        for tag in TAGS:
            g = arrs[tag]
            m = (np.abs(g[:, 0]) <= LIM) & (np.abs(g[:, 1]) <= LIM)
            H = np.histogram2d(
                g[m, 1], g[m, 0],
                bins=BINS, range=[[-LIM, LIM], [-LIM, LIM]])[0]
            hists[(name, tag)] = H / len(res) * 100.0
        np.savez_compressed(cfile, ns=ns[name],
                            **{t: hists[(name, t)] for t in TAGS})
        log.info(f'缓存写入 {cfile.name}')

    # ---- 统计落 CSV ----
    edges_ = np.linspace(-LIM, LIM, BINS + 1)
    centers_ = (edges_[:-1] + edges_[1:]) / 2
    def _stat(w):
        w = w / w.sum()
        mean = (w * centers_).sum()
        std = np.sqrt((w * (centers_ - mean) ** 2).sum())
        cdf = np.cumsum(w)
        q = lambda p: centers_[min(int(np.searchsorted(cdf, p)), BINS - 1)]
        return mean, q(0.5), std, q(0.05), q(0.95)
    csv = _here / 'gaze_distribution_stats.csv'
    with open(csv, 'w') as f:
        f.write('dataset,frame,n,pitch_mean,pitch_median,pitch_std,pitch_p5,pitch_p95,'
                'yaw_mean,yaw_median,yaw_std,yaw_p5,yaw_p95\n')
        for name, _ in plan:
            for tag in TAGS:
                H = hists[(name, tag)]
                if not H.any():
                    continue
                cnt = H / 100.0 * ns[name]
                pm, pmd, ps, p5, p95 = _stat(cnt.sum(axis=0))
                ym, ymd, ys, y5, y95 = _stat(cnt.sum(axis=1))
                f.write(f'{name},{tag},{ns[name]},'
                        + ','.join(f'{v:.4f}' for v in
                                   (pm, pmd, ps, p5, p95, ym, ymd, ys, y5, y95)) + '\n')
    log.info(f'输出 {csv}')

    def pool(H):
        n = H.shape[0] // 2
        return H.reshape(n, 2, n, 2).sum(axis=(1, 3))

    def panel_norm(H, n_samples):
        vmin = NOISE_MIN_SAMPLES * 100.0 / n_samples
        vmax = max(H.max() / 3.0, vmin * 20)
        return LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(5, 4, figsize=(24, 27))
    ROW_TAG = {'CCS': 'CCS gaze', 'HEAD': 'Head pose (norm)',
              'HEAD_RAW': 'Head pose (raw)',
              'HEAD_WORLD': 'Head pose (world)', 'HCS': 'HCS gaze'}
    for j, (name, _) in enumerate(plan):
        for i, tag in enumerate(TAGS):
            ax = axes[i, j]
            H = pool(hists[(name, tag)])
            im = ax.imshow(H.T, origin='lower', cmap=cmap,
                           norm=panel_norm(H, ns[name]),
                           extent=[-LIM, LIM, -LIM, LIM], aspect='equal')
            ax.set_axisbelow(False)
            ax.set_xticks(np.arange(-LIM, LIM + 1, 30))
            ax.set_yticks(np.arange(-LIM, LIM + 1, 30))
            ax.grid(which='major', color='white', lw=0.5)
            ax.set_xticks(np.arange(-LIM, LIM + 1, 15), minor=True)
            ax.set_yticks(np.arange(-LIM, LIM + 1, 15), minor=True)
            ax.grid(which='minor', color='white', lw=0.2, alpha=0.5)
            ax.tick_params(which='minor', length=0)
            title = f'{name}  {ROW_TAG[tag]}'
            if tag in ('HEAD', 'HEAD_RAW'):
                title += ' =world' if name == 'xgaze' else ' =Euler norm/raw'
            ax.set_title(title, fontsize=13)
            ax.set_xlabel('Yaw / Azim (deg)')
            ax.set_ylabel('Pitch / Elev (deg)')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                         label='density (% per bin)')

    fig.suptitle('Gaze & head-pose distribution — ours-without-roll v3 '
                 '(fixed_forward=True)  rows: CCS / Head norm / Head raw / '
                 'Head world / HCS — head norm == raw (v3), xgaze head rows '
                 'use world frame, EVE world = basler cam',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = _here / 'gaze_distribution_noroll.png'
    fig.savefig(out, dpi=200)
    log.info(f'输出 {out}')


if __name__ == '__main__':
    main()
