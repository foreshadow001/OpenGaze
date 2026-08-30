"""四数据集 gaze 分布热力图（specific-face-model 链路，2026-08-30 v2 提速版）

与 sample_all_datasets.py 同几何链路（标准系模型 + DLT/PnP + 归一化），但
按「frame」为处理单位、每被试取 K 个 frame、不读任何图像（纯几何，dummy 图）：
- xgaze/eve：每个 frame 全部相机入样（与 v1 分布图逐 (frame,cam) 口径一致），
  DLT/Kabsch 头姿每 frame 一次；
- gc/mpii：逐帧 6 点 PnP。
每数据集采样 ~2000+ frame，画 gaze pitch×yaw 2D 直方图热力图：
两行 = CCS（归一化相机系）/ HCS（头架系，与归一化无关），
一个数据集一列；±120°，风格与 insightface_224 分布图一致
（共用色轴 深蓝→浅蓝→浅绿→浅黄→红；深蓝底 + 白色网格 30°主/15°细 +
方格等比 + 横轴 yaw、纵轴 pitch）。

输出: 本目录 gaze_distribution_specific.png
用法（仓库根目录）: python preprocess/zhang2015-specific-face-model/viz/distribution/gaze_distribution.py
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
from matplotlib.colors import LogNorm, LinearSegmentedColormap
import numpy as np
from tqdm import tqdm

_PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_PROJECT / 'preprocess/zhang2015-specific-face-model/get_face_model'))

import face_model_core as core
from utils.logger import get_logger
from utils.normalization import estimateHeadPose, normalizeData_face, vector_to_angles

log = get_logger('preprocess.specific_face_model.gaze_dist')

_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    'sample_all', _here.parent / 'sample_all_datasets.py')
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)
# 官方 GC 预处理器（dot→CCS→相机系唯一实现，避免手抄漂移）
_gc_spec = importlib.util.spec_from_file_location(
    'gc_pre', _PROJECT / 'preprocess/zhang2015-insightface/gazecapture/preprocessor.py')
gc_pre = importlib.util.module_from_spec(_gc_spec)
_gc_spec.loader.exec_module(gc_pre)

DUMMY = np.zeros((32, 32, 3), np.uint8)
LIM, BINS = 120, 480        # ±120°，0.5° 一格
NOISE_MIN_SAMPLES = 3        # 可见门限：池化格内 ≥3 个样本（自适应各面板样本量）


def angles(model, rvec, tvec, gp, K):
    """(ccs(p,y), hcs(p,y)) 度数——归一化在 dummy 图上完成（warp 结果弃用）"""
    _, hr, gc = normalizeData_face(
        DUMMY, model, rvec, tvec, gp, K, fixed_forward=False)[:3]
    hR = cv2.Rodrigues(hr)[0]
    return (np.degrees(vector_to_angles(gc.ravel())),
            np.degrees(vector_to_angles((hR.T @ gc).ravel())))


# ------------------------------------------------------------ XGaze
def sample_xgaze(rng, k_frame=300):
    LM = Path('/media/yanglinxuan/ylx/xgaze_insightface_224')
    CAL = Path('/media/yanglinxuan/Expansion/xgaze_raw/calibration/cam_calibration')
    ANN = Path('/media/yanglinxuan/Expansion/xgaze_raw/data/annotation_train')
    FM = Path('/media/yanglinxuan/sfm/xgaze_specific_face_model/face_models')

    KS, DIST, ROT, TR = {}, {}, {}, {}
    for c in range(18):
        fs = cv2.FileStorage(str(CAL / f'cam{c:02d}.xml'), cv2.FILE_STORAGE_READ)
        KS[c] = fs.getNode('Camera_Matrix').mat()
        DIST[c] = fs.getNode('Distortion_Coefficients').mat()
        ROT[c] = fs.getNode('cam_rotation').mat()
        TR[c] = fs.getNode('cam_translation').mat().reshape(3, 1)
        fs.release()

    out = []
    subs = sorted(p.stem for p in LM.glob('subject*.h5'))
    for sid in tqdm(subs, desc='xgaze', unit='subj'):
        model = np.loadtxt(FM / sid / 'true6_canonical.txt')
        ann = {}
        with open(ANN / f'{sid}.csv') as f:
            for line in f:
                p = line.strip().split(',')
                ann[(p[0], p[1])] = np.array(
                    [float(p[4]), float(p[5]), float(p[6])]).reshape(3, 1)
        with h5py.File(LM / f'{sid}.h5', 'r') as f:
            fr = f['frame_index'][:].ravel()
            ci = f['cam_index'][:].ravel()
            lm = f['facial_landmarks_2d'][:]
        by_frame = {}
        for r in range(len(fr)):
            by_frame.setdefault(int(fr[r]), []).append((int(ci[r]), r))
        frames = sorted(by_frame)
        picked = rng.choice(frames, size=min(k_frame, len(frames)), replace=False)
        for fidx in picked:
            rows = by_frame[fidx]
            rays, pv = [], []
            for c, r in rows:
                if c >= 10:
                    continue
                lm_n = cv2.undistortPoints(
                    lm[r].astype(np.float64).reshape(-1, 1, 2),
                    KS[c], DIST[c]).reshape(-1, 2)
                rays.append(lm_n[core.IDX6])
                pv.append(np.concatenate([cv2.Rodrigues(ROT[c])[0].ravel(),
                                          TR[c].ravel()]))
            if len(rays) < 6:
                continue
            X_w = core.triangulate(np.stack(rays), np.stack(pv), n_points=6)
            R_head, t_head = core.kabsch(model, X_w)
            for c, _ in rows:                       # 全部相机入样（v1 口径）
                gp = ann.get((f'frame{fidx:04d}', f'cam{c:02d}.JPG'))
                if gp is None:
                    continue
                rvec = cv2.Rodrigues(ROT[c] @ R_head)[0]
                tvec = ROT[c] @ t_head.reshape(3, 1) + TR[c]
                out.append((model, rvec, tvec, gp, KS[c]))
    return out


# ------------------------------------------------------------ EVE
def sample_eve(rng, k_frame=110):
    LM_ROOT = Path('/media/yanglinxuan/zyx/EVE_dataset/eve_dataset/landmarks')
    RAW = Path('/media/yanglinxuan/zyx/EVE_dataset/eve_dataset')
    FM = Path('/media/yanglinxuan/sfm/eve_specific_face_model/face_models')

    subs = [(sp, p.stem) for sp in ('train', 'test')
            for p in sorted(LM_ROOT.joinpath(sp).glob('*.h5'))]
    out = []
    for sp, subj in tqdm(subs, desc='eve', unit='subj'):
        mpath = FM / subj / 'true6_canonical.txt'
        if not mpath.is_file():
            continue
        model = np.loadtxt(mpath)
        with h5py.File(LM_ROOT / sp / f'{subj}.h5', 'r') as f:
            fr = f['frame_index'][:].ravel()
            ci = f['cam_index'][:].ravel()
            st = f['step_index'][:].ravel()
            lm = f['facial_landmarks_2d'][:]
            cameras = json.loads(f.attrs['cameras'])
            steps = json.loads(f.attrs['steps'])
        by_sync = {}
        for r in range(len(fr)):
            c, raw_f, si = int(ci[r]), int(fr[r]), int(st[r])
            by_sync.setdefault((raw_f // 2 if c == 0 else raw_f, si),
                               []).append((c, r, raw_f, si))
        groups = [v for v in by_sync.values() if len(set(x[0] for x in v)) >= 3]
        if not groups:
            continue
        picked = rng.choice(len(groups), size=min(k_frame, len(groups)),
                            replace=False)
        for gi in picked:
            rows_raw = groups[gi]
            step_name = steps[rows_raw[0][3]]
            Ks, Rs, ts, ann_c = {}, {}, {}, {}
            ok = True
            for c, r, raw_f, _si in rows_raw:
                p_h5 = RAW / subj / step_name / f'{cameras[c]}.h5'
                if not p_h5.is_file():
                    ok = False
                    break
                with h5py.File(p_h5, 'r') as f:
                    if c not in Ks:
                        Ks[c] = np.array(f['camera_matrix'], dtype=float)
                        T = np.array(f['camera_transformation'], dtype=float)
                        Rs[c] = T[:3, :3]
                        ts[c] = T[:3, 3].reshape(3, 1)
                    if raw_f >= len(f['face_PoG_tobii/validity']) or \
                            not f['face_PoG_tobii/validity'][raw_f]:
                        continue
                    # 官方 PoG 直算：屏幕 px → 相机系 3D 注视点（§4，
                    # 各相机标注为同一 tobii 流分发，跨相机一致 ~0.04°）
                    PoG = np.array(f['face_PoG_tobii/data'][raw_f])
                    mmpp = np.array(f['millimeters_per_pixel'], dtype=float)
                    ann_c[c] = (T @ np.array(
                        [PoG[0] * mmpp[0], PoG[1] * mmpp[1], 0., 1.]))[:3]
            if not ok or len(ann_c) < 3:
                continue
            rays, pv = [], []
            for c, r, _rf, _si in rows_raw:
                if c not in Ks:
                    continue
                lm_n = cv2.undistortPoints(
                    lm[r].astype(np.float64).reshape(-1, 1, 2),
                    Ks[c], None).reshape(-1, 2)
                rays.append(lm_n[core.IDX6])
                pv.append(np.concatenate([cv2.Rodrigues(Rs[c])[0].ravel(),
                                          ts[c].ravel()]))
            if len(rays) < 3:
                continue
            X_w = core.triangulate(np.stack(rays), np.stack(pv), n_points=6)
            R_head, t_head = core.kabsch(model, X_w)
            for c, gp_dir in ann_c.items():
                tv = Rs[c] @ t_head.reshape(3, 1) + ts[c]
                gp = gp_dir.reshape(3, 1)
                out.append((model, cv2.Rodrigues(Rs[c] @ R_head)[0], tv, gp,
                            Ks[c]))
    return out


# ------------------------------------------------------------ GazeCapture
def sample_gc(rng, k_frame=500, n_sess=60):
    LM = Path('/media/yanglinxuan/zyx/GazeCapture/landmarks')
    RAW = Path('/media/yanglinxuan/zyx/GazeCapture')
    CAL = Path('/media/yanglinxuan/zyx/GazeCapture/calibration')

    sess = [(sp, p.stem) for sp in ('train', 'test')
            for p in sorted(LM.joinpath(sp).glob('*.h5'))]
    out = []
    for sp, s in tqdm(sess, desc='gazecapture', unit='sess'):
        rec = RAW / s
        try:
            device = json.load(open(rec / 'info.json'))['DeviceName']
            dot = json.load(open(rec / 'dotInfo.json'))
            pos = {int(n.split('.')[0]): i for i, n in
                   enumerate(json.load(open(rec / 'frames.json')))}
        except Exception:
            continue
        slug = device.lower().replace(' ', '-')
        cals = {}
        for (w, h) in ((480, 640), (640, 480)):
            cal = CAL / f'{slug}_{w}x{h}.xml'
            if cal.is_file():
                fs = cv2.FileStorage(str(cal), cv2.FILE_STORAGE_READ)
                cals[(w, h)] = (fs.getNode('Camera_Matrix').mat(),
                                fs.getNode('Distortion_Coefficients').mat())
                fs.release()
        if not cals:
            continue
        with h5py.File(LM / sp / f'{s}.h5', 'r') as f:
            fr = f['frame_index'][:].ravel()
            ori = f['orientation'][:].ravel()
            lm = f['facial_landmarks_2d'][:]
        idx = rng.choice(len(fr), size=min(k_frame, len(fr)), replace=False)
        for i in idx:
            fidx, o, lm106 = int(fr[i]), int(ori[i]), lm[i]
            pi = pos.get(fidx)
            if pi is None or dot['DotNum'][pi] == -1:
                continue
            w, h = (480, 640) if o in (1, 2) else (640, 480)
            if (w, h) not in cals:
                continue
            K, dist = cals[(w, h)]
            # 官方链路直调：dot→CCS（过 invalid_dot 门）→相机系
            xc, yc = dot['XCam'][pi], dot['YCam'][pi]
            ccs_x, ccs_y = gc_pre._dot_to_ccs_mm(o, xc, yc)
            if ccs_y <= 0:
                continue                     # invalid_dot（朝向过渡帧噪声）
            gp = np.array(gc_pre._gaze_point_cam(o, ccs_x, ccs_y))
            rvec, tvec = estimateHeadPose(
                lm106[m.IDX6].reshape(6, 1, 2).astype(float), m.GEN_XE6, K, dist)
            out.append((m.GEN_XE6, rvec, tvec, gp.reshape(3, 1), K))
        if len(out) >= n_sess * k_frame:
            break
    return out


# ------------------------------------------------------------ MPIIFaceGaze
def sample_mpii(rng, k_frame=4000):
    LM = Path('/media/yanglinxuan/zyx/MPIIFaceGaze/landmarks')
    RAW = Path('/media/yanglinxuan/zyx/MPIIFaceGaze')
    import scipy.io as sio

    out = []
    for subj in tqdm(sorted(p.stem for p in LM.glob('*.h5')), desc='mpii',
                     unit='subj'):
        with h5py.File(LM / f'{subj}.h5', 'r') as f:
            lm = f['facial_landmarks_2d'][:]
            names = [n.decode() if isinstance(n, bytes) else str(n)
                     for n in f['image_name'][:]]
            days = f['day_index'][:].ravel()
        mat = sio.loadmat(RAW / subj / 'Calibration' / 'Camera.mat')
        K = np.array(mat['cameraMatrix'], dtype=float)
        dist = np.array(mat['distCoeffs'], dtype=float).ravel()
        gpt = {}
        with open(RAW / subj / f'{subj}.txt') as f:
            for line in f:
                p = line.split()
                gpt[p[0]] = np.array([float(p[24]), float(p[25]),
                                      float(p[26])]).reshape(3, 1)
        idx = rng.choice(len(lm), size=min(k_frame, len(lm)), replace=False)
        for i in idx:
            day, fname = int(days[i]), names[i]
            gp = gpt.get(f'day{day:02d}/{fname}')
            if gp is None:
                continue
            rvec, tvec = estimateHeadPose(
                lm[i][m.IDX6].reshape(6, 1, 2).astype(float), m.GEN_XE6, K, dist)
            out.append((m.GEN_XE6, rvec, tvec, gp, K))
    return out


# ------------------------------------------------------------ 出图
def main():
    rng = np.random.default_rng(7)
    plan = [('xgaze', sample_xgaze), ('eve', sample_eve),
            ('gazecapture', sample_gc), ('mpiifacegaze', sample_mpii)]
    # 色带对齐参考图：turbo（暗蓝→蓝→青→绿→黄→橙→红），底端融入深蓝黑背景
    BG = '#101d4a'                                   # 零/噪声计数 = 深蓝底（稍亮）
    cmap = matplotlib.colormaps['turbo'].copy()
    cmap.set_under(BG)                               # <vmin（噪声门下）
    cmap.set_bad(BG)                                 # LogNorm 把 0 计数归为 bad，须同色

    # 逐数据集缓存（gd_cache_<ds>.npz）：GAZE_DIST_REFRESH=1|all 全刷，
    # 逗号列表（如 GAZE_DIST_REFRESH=xgaze）只刷指定数据集，其余秒级走缓存
    refresh = os.environ.get('GAZE_DIST_REFRESH', '')
    refresh_all = refresh.lower() in ('1', 'all')
    refresh_set = set() if refresh_all else {x for x in refresh.split(',') if x}
    hists, ns = {}, {}

    # 旧单体缓存一次性迁移为逐数据集缓存
    legacy = _here / 'gaze_distribution_cache.npz'
    if legacy.is_file():
        z = np.load(legacy)
        ns_all, hs_all = {}, {}
        for k in z.files:
            if k.startswith('ns|'):
                ns_all[k[3:]] = int(z[k])
            else:
                n_, t_ = k.split('|')
                hs_all[(n_, t_)] = z[k]
        for name, _fn in plan:
            c = _here / f'gd_cache_{name}.npz'
            if not c.is_file() and (name, 'CCS') in hs_all and name in ns_all:
                np.savez_compressed(c, ns=ns_all[name], CCS=hs_all[(name, 'CCS')],
                                    HCS=hs_all[(name, 'HCS')])
        legacy.rename(_here / 'gaze_distribution_cache.npz.migrated')
        log.info('旧单体缓存已迁移为逐数据集缓存')

    for name, fn in plan:
        cfile = _here / f'gd_cache_{name}.npz'
        if not refresh_all and name not in refresh_set and cfile.is_file():
            z = np.load(cfile)
            ns[name] = int(z['ns'])
            for tag in ('CCS', 'HCS'):
                hists[(name, tag)] = z[tag]
            log.info(f'缓存命中 {name}（n={ns[name]:,}）')
            continue
        items = fn(rng)
        res = [angles(*it) for it in
               tqdm(items, desc=f'{name} 角度', unit='样本', leave=False)]
        ns[name] = len(res)
        log.info(f'{name}: {ns[name]} 样本（(frame,cam) 口径）')
        for g, tag in [(np.array([r[0] for r in res]), 'CCS'),
                       (np.array([r[1] for r in res]), 'HCS')]:
            H = np.histogram2d(
                np.clip(g[:, 1], -LIM, LIM), np.clip(g[:, 0], -LIM, LIM),
                bins=BINS, range=[[-LIM, LIM], [-LIM, LIM]])[0]
            # 密度归一（% 每格）：与样本量无关，同色 = 同概率质量
            hists[(name, tag)] = H / len(res) * 100.0
        np.savez_compressed(cfile, ns=ns[name], CCS=hists[(name, 'CCS')],
                            HCS=hists[(name, 'HCS')])
        log.info(f'缓存写入 {cfile.name}')
    # ---- 统计落 CSV（由缓存直方图加权计算） ----
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
            for tag in ('CCS', 'HCS'):
                H = hists[(name, tag)]              # (n_yaw, n_pitch) 密度
                cnt = H / 100.0 * ns[name]
                pm, pmd, ps, p5, p95 = _stat(cnt.sum(axis=0))   # pitch 边际
                ym, ymd, ys, y5, y95 = _stat(cnt.sum(axis=1))   # yaw 边际
                f.write(f'{name},{tag},{ns[name]},'
                        + ','.join(f'{v:.4f}' for v in
                                   (pm, pmd, ps, p5, p95, ym, ymd, ys, y5, y95)) + '\n')
    log.info(f'输出 {csv}')

    # 显示时 2×2 池化到 1°：摊薄型（xgaze CCS 带密度 ~0.03%/0.5°格，压在
    # VMIN 阈值边上导致空白）密度×4 进入可见区；孤立单点仍低于阈值不糊
    def pool(H):
        n = H.shape[0] // 2
        return H.reshape(n, 2, n, 2).sum(axis=(1, 3))

    # 每个热力图独立色轴：vmin = NOISE_MIN_SAMPLES 个样本的密度（自适应，
    # xgaze 平坦分布的峰值只有紧簇型的 1/200，固定阈值会整图落底）；
    # vmax = 峰值密度的 1/3（红 = 核心区）
    def panel_norm(H, n_samples):
        vmin = NOISE_MIN_SAMPLES * 100.0 / n_samples
        vmax = max(H.max() / 3.0, vmin * 20)
        return LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(2, 4, figsize=(24, 12.5))
    for j, (name, _) in enumerate(plan):
        for i, tag in enumerate(['CCS', 'HCS']):
            ax = axes[i, j]
            H = pool(hists[(name, tag)])
            im = ax.imshow(H.T, origin='lower', cmap=cmap,
                           norm=panel_norm(H, ns[name]),
                           extent=[-LIM, LIM, -LIM, LIM], aspect='equal')
            ax.set_axisbelow(False)
            # 白色网格：30° 主格 + 15° 细格（细线）
            ax.set_xticks(np.arange(-LIM, LIM + 1, 30))
            ax.set_yticks(np.arange(-LIM, LIM + 1, 30))
            ax.grid(which='major', color='white', lw=0.5)
            ax.set_xticks(np.arange(-LIM, LIM + 1, 15), minor=True)
            ax.set_yticks(np.arange(-LIM, LIM + 1, 15), minor=True)
            ax.grid(which='minor', color='white', lw=0.2, alpha=0.5)
            ax.tick_params(which='minor', length=0)
            ax.set_title(f'{name}  {tag}  ({ns[name]:,} samples)', fontsize=13)
            ax.set_xlabel('Yaw (deg)')
            ax.set_ylabel('Pitch (deg)')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                         label='density (% per bin)')

    fig.suptitle('Gaze distribution — specific-face-model chain (canonical models) '
                 '— top: CCS (normalized cam), bottom: HCS (head frame)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = _here / 'gaze_distribution_specific.png'
    fig.savefig(out, dpi=200)
    log.info(f'输出 {out}')


if __name__ == '__main__':
    main()
