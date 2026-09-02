"""v1（insightface_224）四数据集 gaze & head pose 分布热力图（2026-08-31，v2 风格）

从**原始数据**出发按 v1 链路（通用 gen6 6 点 PnP + 各数据集官方视线链）采样：
- 与 v2 版同采样量（xgaze 300帧/人, EVE 110组/人, GC 600sess×500帧, MPII 4000/人）；
- 2 行 × 4 列热力图：CCS gaze / Head pose（gen6 旧口径，is_true6=False）；
- turbo 色带 + 深蓝底 + 白色 30°主/15°细网格 + 方格等比 + 独立对数色柱。

GC 质量门（appleFace/eye IsValid）与 v1 预处理器同步启用。
同时刷新 gaze_distribution_stats.csv。

输出: 本目录 gaze_distribution_specific.png / gaze_distribution_stats.csv
用法（仓库根目录）:
  /ssd/conda/envs/yanglinxuan/opengaze/bin/python \
  preprocess/zhang2015-insightface/viz/gaze_distribution/gaze_distribution.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from tqdm import tqdm

_PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_PROJECT / 'preprocess/zhang2015-specific-face-model/get_face_model'))

import face_model_core as core                              # noqa: E402
from utils.logger import get_logger                         # noqa: E402
from utils.normalization import (estimateHeadPose, head_pose_angles,  # noqa: E402
                                 normalizeData_face, vector_to_angles)

log = get_logger('preprocess.insightface.viz.gaze_dist')

# GC 官方链（唯一实现）
_gc_spec = importlib.util.spec_from_file_location(
    'gc_pre',
    _PROJECT / 'preprocess/zhang2015-insightface/gazecapture/preprocessor.py')
gc_pre = importlib.util.module_from_spec(_gc_spec)
_gc_spec.loader.exec_module(gc_pre)

HERE = Path(__file__).resolve().parent
DUMMY = np.zeros((32, 32, 3), np.uint8)
LIM, BINS = 120, 480        # ±120°，0.5° 一格
NOISE_MIN_SAMPLES = 3
IDX6 = [35, 39, 89, 93, 78, 84]

DATASETS = ['xgaze', 'eve', 'mpiifacegaze', 'gazecapture']
ARMS = ['CCS', 'HEAD']


def _angles(model, rvec, tvec, gp, K):
    """一次归一化同时得 (ccs(p,y), head(p,y)) 度数（gen6 旧口径）"""
    _, hr, gc = normalizeData_face(
        DUMMY, model, rvec, tvec, gp, K, fixed_forward=False)[:3]
    hR = cv2.Rodrigues(hr)[0]
    return (np.degrees(vector_to_angles(gc.ravel())),
            head_pose_angles(hR, is_true6=False))


# ------------------------------------------------------------ XGaze
def sample_xgaze(rng, k_frame=300):
    LM = Path('/media/yanglinxuan/ylx/xgaze_insightface_224')
    CAL = Path('/media/yanglinxuan/Expansion/xgaze_raw/calibration/cam_calibration')
    ANN = Path('/media/yanglinxuan/Expansion/xgaze_raw/data/annotation_train')
    GEN6 = np.loadtxt(
        _PROJECT / 'preprocess/zhang2015-insightface/face_model_xgaze.txt'
    )[[20, 23, 26, 29, 15, 19], :]

    KS, DIST = {}, {}
    for c in range(18):
        fs = cv2.FileStorage(str(CAL / f'cam{c:02d}.xml'), cv2.FILE_STORAGE_READ)
        KS[c] = fs.getNode('Camera_Matrix').mat()
        DIST[c] = fs.getNode('Distortion_Coefficients').mat()
        fs.release()

    out = []
    for sid in tqdm(sorted(p.stem for p in LM.glob('subject*.h5')),
                    desc='xgaze', unit='subj'):
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
        picked = rng.choice(frames, size=min(k_frame, len(frames)),
                            replace=False)
        for fidx in picked:
            rows = by_frame[fidx]
            for c, r in rows:
                gp = ann.get((f'frame{fidx:04d}', f'cam{c:02d}.JPG'))
                if gp is None:
                    continue
                try:
                    rvec, tvec = estimateHeadPose(
                        lm[r][core.IDX6].reshape(6, 1, 2).astype(float),
                        GEN6, KS[c], DIST[c])
                except cv2.error:
                    continue
                out.append(_angles(GEN6, rvec, tvec, gp, KS[c]))
    return out


# ------------------------------------------------------------ GazeCapture
def sample_gc(rng, k_frame=500, n_sess=600):
    LM = Path('/media/yanglinxuan/zyx/GazeCapture/landmarks')
    RAW = Path('/media/yanglinxuan/zyx/GazeCapture')
    CAL = Path('/media/yanglinxuan/zyx/GazeCapture/calibration')
    GEN6 = np.loadtxt(
        _PROJECT / 'preprocess/zhang2015-insightface/face_model_xgaze.txt'
    )[[20, 23, 26, 29, 15, 19], :]

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
            face = json.load(open(rec / 'appleFace.json'))
            leye = json.load(open(rec / 'appleLeftEye.json'))
            reye = json.load(open(rec / 'appleRightEye.json'))
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
            if not (face['IsValid'][pi] and leye['IsValid'][pi]
                    and reye['IsValid'][pi]):
                continue                 # 官方四条件质量门
            w, h = (480, 640) if o in (1, 2) else (640, 480)
            if (w, h) not in cals:
                continue
            K, dist = cals[(w, h)]
            xc, yc = dot['XCam'][pi], dot['YCam'][pi]
            ccs_x, ccs_y = gc_pre._dot_to_ccs_mm(o, xc, yc)
            if ccs_y <= 0:
                continue
            gp = np.array(gc_pre._gaze_point_cam(o, ccs_x, ccs_y)).reshape(3, 1)
            try:
                rvec, tvec = estimateHeadPose(
                    lm106[core.IDX6].reshape(6, 1, 2).astype(float),
                    GEN6, K, dist)
            except cv2.error:
                continue
            out.append(_angles(GEN6, rvec, tvec, gp, K))
        if len(out) >= n_sess * k_frame:
            break
    return out


# ------------------------------------------------------------ MPIIFaceGaze
def sample_mpii(rng, k_frame=4000):
    LM = Path('/media/yanglinxuan/zyx/MPIIFaceGaze/landmarks')
    RAW = Path('/media/yanglinxuan/zyx/MPIIFaceGaze')
    GEN6 = np.loadtxt(
        _PROJECT / 'preprocess/zhang2015-insightface/face_model_xgaze.txt'
    )[[20, 23, 26, 29, 15, 19], :]
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
            try:
                rvec, tvec = estimateHeadPose(
                    lm[i][core.IDX6].reshape(6, 1, 2).astype(float),
                    GEN6, K, dist)
            except cv2.error:
                continue
            out.append(_angles(GEN6, rvec, tvec, gp, K))
    return out


# ------------------------------------------------------------ EVE
def sample_eve(rng, k_frame=110):
    LM_ROOT = Path('/media/yanglinxuan/zyx/EVE_dataset/eve_dataset/landmarks')
    RAW = Path('/media/yanglinxuan/zyx/EVE_dataset/eve_dataset')
    GEN6 = np.loadtxt(
        _PROJECT / 'preprocess/zhang2015-insightface/face_model_xgaze.txt'
    )[[20, 23, 26, 29, 15, 19], :]

    subs = [(sp, p.stem) for sp in ('train', 'test')
            for p in sorted(LM_ROOT.joinpath(sp).glob('*.h5'))]
    out = []
    for sp, subj in tqdm(subs, desc='eve', unit='subj'):
        with h5py.File(LM_ROOT / sp / f'{subj}.h5', 'r') as f:
            fr = f['frame_index'][:].ravel()
            ci = f['cam_index'][:].ravel()
            st = f['step_index'][:].ravel()
            lm = f['facial_landmarks_2d'][:]
            cameras = json.loads(f.attrs['cameras'])
            steps = json.loads(f.attrs['steps'])
        Ks = {}
        for c, cam_name in enumerate(cameras):
            for step in steps:
                p = RAW / subj / step / f'{cam_name}.h5'
                if p.is_file():
                    with h5py.File(p, 'r') as f:
                        Ks[c] = np.array(f['camera_matrix'], dtype=float)
                    break
        by_sync = {}
        for r in range(len(fr)):
            c, raw_f = int(ci[r]), int(fr[r])
            by_sync.setdefault((raw_f // 2 if c == 0 else raw_f, int(st[r])),
                               []).append((c, r, raw_f, int(st[r])))
        groups = list(by_sync.values())
        if not groups:
            continue
        picked = rng.choice(len(groups), size=min(k_frame, len(groups)),
                            replace=False)
        for gi in picked:
            rows_raw = groups[gi]
            step_name = steps[rows_raw[0][3]]
            for c, r, raw_f, _ in rows_raw:
                if c not in Ks:
                    continue
                p_h5 = RAW / subj / step_name / f'{cameras[c]}.h5'
                if not p_h5.is_file():
                    continue
                with h5py.File(p_h5, 'r') as f:
                    if raw_f >= len(f['face_PoG_tobii/validity']) or \
                            not f['face_PoG_tobii/validity'][raw_f]:
                        continue
                    PoG = np.array(f['face_PoG_tobii/data'][raw_f])
                    mmpp = np.array(f['millimeters_per_pixel'], dtype=float)
                    T = np.array(f['camera_transformation'], dtype=float)
                gp = (T @ np.array([PoG[0] * mmpp[0], PoG[1] * mmpp[1],
                                    0., 1.]))[:3].reshape(3, 1)
                try:
                    rvec, tvec = estimateHeadPose(
                        lm[r][core.IDX6].reshape(6, 1, 2).astype(float),
                        GEN6, Ks[c], None)
                except cv2.error:
                    continue
                out.append(_angles(GEN6, rvec, tvec, gp, Ks[c]))
    return out


# ------------------------------------------------------------ 出图
def main():
    rng = np.random.default_rng(7)
    BG = '#101d4a'
    cmap = matplotlib.colormaps['turbo'].copy()
    cmap.set_under(BG)
    cmap.set_bad(BG)
    ROW_TAG = {'CCS': 'CCS gaze', 'HEAD': 'Head pose'}
    plan = [('xgaze', sample_xgaze), ('eve', sample_eve),
            ('mpiifacegaze', sample_mpii), ('gazecapture', sample_gc)]

    # 逐数据集缓存（gd_cache_<ds>.npz）：GAZE_DIST_REFRESH=1|all 全刷，
    # 逗号列表只刷指定数据集
    refresh = __import__('os').environ.get('GAZE_DIST_REFRESH', '')
    refresh_all = refresh.lower() in ('1', 'all')
    refresh_set = set() if refresh_all else {x for x in refresh.split(',') if x}
    hists, ns = {}, {}
    for name, fn in plan:
        cfile = HERE / f'gd_cache_{name}.npz'
        if not refresh_all and name not in refresh_set and cfile.is_file():
            z = np.load(cfile)
            ns[name] = int(z['ns'])
            for tag in ARMS:
                hists[(name, tag)] = z[tag]
            log.info(f'缓存命中 {name}（n={ns[name]:,}）')
            continue
        res = fn(rng)
        ns[name] = len(res)
        log.info(f'{name}: {len(res):,} 样本')
        for arr, tag in [(np.array([r[0] for r in res]), 'CCS'),
                         (np.array([r[1] for r in res]), 'HEAD')]:
            m = (np.abs(arr[:, 0]) <= LIM) & (np.abs(arr[:, 1]) <= LIM)
            H = np.histogram2d(arr[m, 1], arr[m, 0], bins=BINS,
                               range=[[-LIM, LIM], [-LIM, LIM]])[0]
            hists[(name, tag)] = H / max(len(arr), 1) * 100.0
        np.savez_compressed(cfile, ns=ns[name],
                            **{t: hists[(name, t)] for t in ARMS})
        log.info(f'缓存写入 {cfile.name}')

    def pool(H):
        n = H.shape[0] // 2
        return H.reshape(n, 2, n, 2).sum(axis=(1, 3))

    def panel_norm(H, n):
        vmin = NOISE_MIN_SAMPLES * 100.0 / max(n, 1)
        vmax = max(H.max() / 3.0, vmin * 20)
        return LogNorm(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(24, 22))
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1])
    curve_axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])]
    COLORS = {'xgaze': 'tab:blue', 'eve': 'tab:green',
              'mpiifacegaze': '#DAA520', 'gazecapture': 'tab:red'}
    from scipy.ndimage import gaussian_filter1d
    ce480 = (np.arange(BINS) + 0.5) * (2 * LIM / BINS) - LIM   # 0.5° 格中心
    for ax, tag_lbl, k in [(curve_axes[0], 'Pitch', 0), (curve_axes[1], 'Yaw', 1)]:
        for name, _ in plan:
            H = hists[(name, 'CCS')]
            marginal = H.sum(axis=1 if k == 0 else 0)   # pitch/yaw 边际
            h = gaussian_filter1d(marginal, sigma=0.8)    # 轻平滑（0.4°）
            ax.plot(ce480, h, lw=2.2, color=COLORS[name],
                    label=name.capitalize() if name != 'mpiifacegaze' else 'MPII')
        ax.set_title(f'Gaze {tag_lbl} Distribution (deg, CCS)', fontsize=12)
        ax.set_xlabel(f'{tag_lbl} (deg)')
        ax.set_ylabel('% per degree')
        ax.set_xlim(-LIM, LIM)
        ax.set_xticks(np.arange(-120, 121, 30))
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    for j, (name, _) in enumerate(plan):
        for i, tag in enumerate(ARMS):
            ax = fig.add_subplot(gs[i + 2, j])
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
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                         label='density (% per bin)')
            ax.set_title(f'{name}  {ROW_TAG[tag]}  ({ns[name]:,} samples)',
                         fontsize=13)
            ax.set_xlabel('Yaw (deg)')
            ax.set_ylabel('Pitch (deg)')

    fig.suptitle('Gaze & head-pose distribution — v1 insightface_224 (gen6 PnP)  '
                 'top: pitch/yaw curves | rows: CCS gaze / Head pose',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = HERE / 'gaze_distribution_specific.png'
    fig.savefig(png, dpi=200)
    log.info(f'输出 {png}')

    csv = HERE / 'gaze_distribution_stats.csv'
    with open(csv, 'w') as f:
        f.write('dataset,frame,n,pitch_mean,pitch_median,pitch_std,'
                'pitch_p5,pitch_p95,yaw_mean,yaw_median,yaw_std,yaw_p5,yaw_p95\n')
        for name, _ in plan:
            for tag in ARMS:
                H = hists[(name, tag)]
                cnt = H / 100.0 * ns[name]
                if not cnt.any():
                    continue
                edges = np.linspace(-LIM, LIM, BINS + 1)
                ce = (edges[:-1] + edges[1:]) / 2

                def stat(w):
                    w = w / max(w.sum(), 1)
                    mean = (w * ce).sum()
                    std = np.sqrt((w * (ce - mean) ** 2).sum())
                    cdf = np.cumsum(w)
                    q = lambda p: ce[min(int(np.searchsorted(cdf, p)), BINS - 1)]
                    return mean, q(0.5), std, q(0.05), q(0.95)
                pm, pmd, ps, p5, p95 = stat(cnt.sum(axis=0))
                ym, ymd, ys, y5, y95 = stat(cnt.sum(axis=1))
                f.write(f'{name},{tag},{ns[name]},'
                        + ','.join(f'{v:.4f}' for v in
                                   (pm, pmd, ps, p5, p95,
                                    ym, ymd, ys, y5, y95)) + '\n')
    log.info(f'输出 {csv}')


if __name__ == '__main__':
    main()
