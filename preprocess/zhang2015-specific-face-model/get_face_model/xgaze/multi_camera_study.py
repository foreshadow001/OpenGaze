"""ETH-XGaze 多相机 vs 单相机人脸 3D 模型重建对比实验（2026-08-28，与 EVE 版协议对齐）

与 EVE 版（../eve/multi_camera_study.py）的差异：
  - 18 台穹顶相机、视角跨度大 → 按用户要求只用「不太极端」的视角：
    逐视图 <60° 过滤 + 相机选择（训练帧中位姿态角 <40° 且 ≥15 视图），与现行建模协议一致；
    所有实验臂（S/S240/F/M/GEN/S_cross）都只在选定相机集上做，确保各臂可比。
  - 帧宇宙跨相机共享（同一 frame_index 各相机同步拍摄）：统一测试集 = 120 帧采样的奇数位，
    S240 训练集需与测试帧索引距离 >2（密集帧率下相邻帧近重复，防泄漏）。
  - 内参/畸变逐相机取 xml（非零畸变，undistortPoints 消除）。
  - S240/S_cross 的源相机 = 每被试中位角最小的相机（最接近正面，EVE 版固定 basler 的对应物）。
  - 额外诊断：选定相机集内 S 模型两两 Kabsch 结构差异（mm），量化相机间观测一致性。

实验臂（同一被试、同一套帧采样）：
  S     单相机逐组（现行方案：三角化 → 剔坏帧 → BA）
  T     消融：仅三角化（全部训练帧，不剔帧不 BA）
  Tk    消融：剔除坏帧后仅三角化（不 BA）——T→Tk→S 拆出剔帧与 BA 的各自贡献
  S240  源相机视图数对照（240 训练视图）——区分「相机多样性」与「帧数多」
  F     模型级融合（各相机 S 模型 Kabsch 对齐取逐点中位，core.canonical_of）
  M     多相机联合（选定相机 × 60 帧观测级合并 BA，逐视图 f_px）
  GEN   通用 50 点模型基线（context）
  S_cross 源相机 S 模型跨相机评估（单相机模型对未见相机的泛化）

输入: /media/yanglinxuan/ylx/xgaze_insightface_224/subject{NNNN}.h5（只读）+ cam_calibration xml
输出: metrics/multi_camera/{per_subject.csv, aggregate.md, pairwise_models_mm.csv}
用法（仓库根目录运行；CPU 即可）:
  .../multi_camera_study.py [-j 12] [-sb 0 -se 80]
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys
import time
import argparse
from pathlib import Path
from multiprocessing import Pool

import cv2
import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # get_face_model/
sys.path.insert(0, str(PROJECT_ROOT))

import face_model_core as core                                        # noqa: E402
from utils.logger import get_logger                                   # noqa: E402

log = get_logger('preprocess.specific_face_model.xgaze.multicam_study')

LM_DIR = '/media/yanglinxuan/ylx/xgaze_insightface_224'
CALIB_DIR = '/media/yanglinxuan/Expansion/xgaze_raw/calibration/cam_calibration'
METRICS_DIR = Path(__file__).resolve().parent / 'metrics' / 'multi_camera'
FACE_MODEL_FILE = Path(__file__).resolve().parents[3] / 'zhang2015-insightface' / 'face_model_xgaze.txt'
GEN_ROWS = [20, 23, 26, 29, 15, 19]

N_SAMPLE = 120          # 每相机采样帧数（与现行建模协议一致）
N_SAMPLE_240 = 480      # S240 臂的帧采样数
LEAK_GUARD = 2          # S240 训练帧与统一测试帧的最小索引距离
GEN6 = None             # worker 内初始化
KS = DISTS = None


def init_worker():
    global GEN6, KS, DISTS
    GEN6 = np.loadtxt(FACE_MODEL_FILE)[GEN_ROWS, :]
    KS, DISTS = {}, {}
    for c in range(18):
        fs = cv2.FileStorage(str(Path(CALIB_DIR) / f'cam{c:02d}.xml'),
                             cv2.FILE_STORAGE_READ)
        KS[c] = fs.getNode('Camera_Matrix').mat()
        DISTS[c] = fs.getNode('Distortion_Coefficients').mat()
        fs.release()


def init_view(lm_px, c, gen6):
    """单帧观测初始化（逐相机 K/dist）：去畸变刚性 28 点 + GEN6 PnP 姿态与角度"""
    lm_n, pose, ang = core.init_view(lm_px.astype(np.float64), KS[c], DISTS[c], gen6)
    return lm_n, pose, ang


# ---------------------------------------------------------------- 多相机 BA
def ba_multicam(lm_t, pv_t, X0, f_pxs, n_points=core.NP):
    """多相机联合 BA：与 core.ba_shared 同构，残差按逐视图 f_px 换算到像素单位
    （soft_l1 鲁棒损失在像素域统一 f_scale=5px）。"""
    NV = len(pv_t)

    def residuals(p):
        X = p[:3 * n_points].reshape(n_points, 3)
        rv = p[3 * n_points:3 * n_points + 3 * NV].reshape(NV, 3)
        tv = p[3 * n_points + 3 * NV:].reshape(NV, 3)
        return ((core.proj(X, rv, tv) - lm_t) * f_pxs[:, None, None]).ravel()

    x0 = np.concatenate([X0.ravel(), pv_t[:, :3].ravel(), pv_t[:, 3:].ravel()])
    from scipy.sparse import lil_matrix
    S = lil_matrix((NV * n_points * 2, len(x0)), dtype=int)
    for v in range(NV):
        for j in range(n_points):
            r0 = (v * n_points + j) * 2
            S[r0:r0 + 2, 3 * j:3 * j + 3] = 1
            S[r0:r0 + 2, 3 * n_points + 3 * v:3 * n_points + 3 * v + 6] = 1
    from scipy.optimize import least_squares
    r = least_squares(residuals, x0, jac_sparsity=S, loss='soft_l1',
                      f_scale=core.F_SCALE_PX, method='trf',
                      xtol=1e-12, ftol=1e-12, max_nfev=core.MAX_NFEV)
    X = r.x[:3 * n_points].reshape(n_points, 3)
    rr = residuals(r.x).reshape(NV, n_points, 2)
    fr_rms = np.sqrt(np.mean(rr ** 2, axis=(1, 2)))
    return X, fr_rms


def model_group_multicam(train_views, f_pxs):
    """多相机版 model_group：固定姿态三角化 → 好帧保留 → 联合 BA（逐视图 f_px）"""
    lm_t = np.stack([v[0] for v in train_views])
    pv_t = np.stack([v[1] for v in train_views])
    X0 = core.triangulate(lm_t, pv_t)
    pr = core.proj(X0, pv_t[:, :3], pv_t[:, 3:])
    fr_rms = np.sqrt(np.mean(((pr - lm_t) * f_pxs[:, None, None]) ** 2, axis=(1, 2)))
    keep = np.argsort(fr_rms)[:max(4, int(len(train_views) * core.KEEP_FRAC))]
    X, ba_rms = ba_multicam(lm_t[keep], pv_t[keep], X0, f_pxs[keep])
    return {'model': X, 'train_rms': float(np.median(ba_rms)),
            'n_kept': int(len(keep)), 'n_train': len(train_views)}


def eval_views(model, views, f_px, idx6_rows=None, gen6=None):
    errs, errs_gen = [], []
    for lm_n, _ in views:
        errs.append(core.pnp_model(model, lm_n, f_px)[2])
        if gen6 is not None:
            errs_gen.append(core.pnp_model(gen6, lm_n[idx6_rows], f_px)[2])
    return (float(np.median(errs)) if errs else float('nan'),
            float(np.median(errs_gen)) if errs_gen else float('nan'))


def iod_of(model):
    return float(np.linalg.norm(model[core.RIGID.index(35)]
                                - model[core.RIGID.index(93)]))


# ---------------------------------------------------------------- 单被试实验
def process_subject(sid):
    gen6 = GEN6
    idx6_rows = [core.RIGID.index(i) for i in core.IDX6]
    subject = f'subject{sid:04d}'
    t0 = time.time()
    with h5py.File(Path(LM_DIR) / f'{subject}.h5', 'r') as f:
        fr_all = f['frame_index'][:].ravel()
        cam_all = f['cam_index'][:].ravel()
        lm_all = f['facial_landmarks_2d'][:]

    # 帧采样（跨相机共享的帧宇宙）→ 统一 train/test
    frames_universe = sorted(set(fr_all.tolist()))
    n_sample = min(N_SAMPLE, len(frames_universe))
    sel = np.array(frames_universe)[np.linspace(0, len(frames_universe) - 1,
                                               n_sample).astype(int)]
    train_frames, test_frames = sorted(sel[0::2].tolist()), sorted(sel[1::2].tolist())

    # 逐 (相机, 帧) 视图初始化
    obs, poses = {}, {}
    cam_angle = {}
    for c in range(18):
        m = (cam_all == c) & np.isin(fr_all, sel)
        angles = []
        for r_, fr_ in zip(np.where(m)[0], fr_all[m]):
            lm_n, pose, ang = init_view(lm_all[r_], c, gen6)
            obs[(c, int(fr_))] = lm_n
            poses[(c, int(fr_))] = pose
            angles.append(ang)
        cam_angle[c] = float(np.median(angles)) if angles else 180.0

    ok_view = lambda vk: np.degrees(np.linalg.norm(poses[vk][:3])) < core.VIEW_ANGLE_MAX

    def views_of(c, frames):
        return [(obs[(c, fr)], poses[(c, fr)]) for fr in frames
                if (c, fr) in obs and ok_view((c, fr))]

    # 相机选择（不太极端：中位角 <40° 且 ≥15 训练视图，与现行协议一致）
    sel_cams = [c for c in range(18)
                if cam_angle[c] < core.CAM_ANGLE_MAX
                and len(views_of(c, train_frames)) >= core.MIN_GROUP_VIEWS]
    if len(sel_cams) < 2:
        return {'subject': subject, 'n_sel': len(sel_cams), 'rows': [],
                'pairs': [], 'sec': time.time() - t0}
    c0 = min(sel_cams, key=lambda c: cam_angle[c])       # 最接近正面的相机

    rows_out, pairs_out = [], []
    fpx = {c: core.f_px_of(KS[c]) for c in sel_cams}
    train_v = {c: views_of(c, train_frames) for c in sel_cams}
    test_v = {c: views_of(c, test_frames) for c in sel_cams}

    # ---- 臂 S：单相机逐组（现行方案）+ 消融臂 T/Tk ----
    s_models = {}
    for c in sel_cams:
        lm_t = np.stack([v[0] for v in train_v[c]])
        pv_t = np.stack([v[1] for v in train_v[c]])
        # T：仅三角化（全部训练帧；train 误差 = 初始姿态回投影残差）
        X0 = core.triangulate(lm_t, pv_t)
        pr = core.proj(X0, pv_t[:, :3], pv_t[:, 3:])
        fr_rms = np.sqrt(np.mean(((pr - lm_t) * fpx[c]) ** 2, axis=(1, 2)))
        te0, _ = eval_views(X0, test_v[c], fpx[c])
        rows_out.append((subject, 'T', f'cam{c:02d}', len(lm_t), len(lm_t),
                         float(np.median(fr_rms)), te0, iod_of(X0)))
        # Tk：剔除坏帧后仅三角化（好帧判定与 core.model_group 完全一致）
        keep = np.argsort(fr_rms)[:max(4, int(len(lm_t) * core.KEEP_FRAC))]
        X0k = core.triangulate(lm_t[keep], pv_t[keep])
        tek, _ = eval_views(X0k, test_v[c], fpx[c])
        rows_out.append((subject, 'Tk', f'cam{c:02d}', len(keep), len(keep),
                         float(np.median(fr_rms[keep])), tek, iod_of(X0k)))
        # S：完整管线
        res = core.model_group(train_v[c], fpx[c])
        s_models[c] = res
        te, _ = eval_views(res['model'], test_v[c], fpx[c])
        rows_out.append((subject, 'S', f'cam{c:02d}', res['n_train'], res['n_kept'],
                         res['train_rms'], te, iod_of(res['model'])))
        # GEN 基线（每相机，同测试帧）
        _, te_gen = eval_views(res['model'], test_v[c], fpx[c], idx6_rows, gen6)
        rows_out.append((subject, 'GEN', f'cam{c:02d}', 0, 0, float('nan'),
                         te_gen, float('nan')))
        # S 模型两两结构差异（诊断）
        for c2 in sel_cams:
            if c2 >= c:
                continue
            R, t = core.kabsch(s_models[c]['model'], s_models[c2]['model'])
            d = np.linalg.norm((R @ s_models[c]['model'].T).T + t
                               - s_models[c2]['model'], axis=1)
            pairs_out.append((subject, f'cam{c:02d}', f'cam{c2:02d}',
                              float(np.sqrt((d ** 2).mean()))))

    # ---- 臂 S240：源相机视图数对照 ----
    n_s = min(N_SAMPLE_240, len(frames_universe))
    sel240 = np.array(frames_universe)[np.linspace(0, len(frames_universe) - 1,
                                                  n_s).astype(int)][0::2]
    guard = np.array(test_frames)
    sel240 = np.array([fr for fr in sel240
                       if len(guard) == 0 or abs(int(fr) - guard).min() > LEAK_GUARD])[:240]
    if len(sel240) >= core.MIN_GROUP_VIEWS:
        rows_idx = np.where(cam_all == c0)[0]
        fr_map = {int(fr): r for r, fr in zip(rows_idx, fr_all[rows_idx])}
        train240 = []
        for fr in sel240:
            if int(fr) in fr_map:
                lm_n, pose, ang = init_view(lm_all[fr_map[int(fr)]], c0, gen6)
                if ang < core.VIEW_ANGLE_MAX:
                    train240.append((lm_n, pose))
        if len(train240) >= core.MIN_GROUP_VIEWS:
            res = core.model_group(train240, fpx[c0])
            te, _ = eval_views(res['model'], test_v[c0], fpx[c0])
            rows_out.append((subject, 'S240', f'cam{c0:02d}', res['n_train'],
                             res['n_kept'], res['train_rms'], te, iod_of(res['model'])))

    # ---- 臂 F：模型级融合 ----
    if len(s_models) >= 2:
        fused = core.canonical_of({c: r['model'] for c, r in s_models.items()})
        for c in sel_cams:
            te, _ = eval_views(fused, test_v[c], fpx[c])
            rows_out.append((subject, 'F', f'cam{c:02d}',
                             sum(r['n_train'] for r in s_models.values()),
                             0, float('nan'), te, iod_of(fused)))

    # ---- 臂 M：多相机联合（选定相机集内观测级合并） ----
    pooled, pooled_fpx = [], []
    for c in sel_cams:
        for v in train_v[c]:
            pooled.append(v)
            pooled_fpx.append(fpx[c])
    if len(pooled) >= core.MIN_GROUP_VIEWS * 2:
        res = model_group_multicam(pooled, np.array(pooled_fpx))
        for c in sel_cams:
            te, _ = eval_views(res['model'], test_v[c], fpx[c])
            rows_out.append((subject, 'M', f'cam{c:02d}', res['n_train'], res['n_kept'],
                             res['train_rms'], te, iod_of(res['model'])))

    # ---- 臂 S_cross：源相机模型跨相机 ----
    for c in sel_cams:
        if c == c0 or not test_v[c]:
            continue
        te, _ = eval_views(s_models[c0]['model'], test_v[c], fpx[c])
        rows_out.append((subject, 'S_cross', f'cam{c:02d}', 0, 0, float('nan'),
                         te, float('nan')))

    return {'subject': subject, 'n_sel': len(sel_cams), 'rows': rows_out,
            'pairs': pairs_out, 'sec': time.time() - t0}


# ---------------------------------------------------------------- 汇总
def aggregate(all_rows):
    agg = {}
    for r in all_rows:
        agg.setdefault((r[1], r[2]), []).append(r)
    lines = ['| arm | cam | n_subj | train_rms_med | test_rms_med | iod_med |',
             '|---|---|---|---|---|---|']
    for (arm, cam), rs in sorted(agg.items()):
        te = [r[6] for r in rs if not np.isnan(r[6])]
        tr = [r[5] for r in rs if not np.isnan(r[5])]
        io = [r[7] for r in rs if not np.isnan(r[7])]
        lines.append('| {} | {} | {} | {} | {} | {} |'.format(
            arm, cam, len(rs),
            f'{np.median(tr):.2f}' if tr else '—',
            f'{np.median(te):.2f}' if te else '—',
            f'{np.median(io):.1f}' if io else '—'))
    lines.append('')
    lines.append('| arm | 全相机 test_rms 中位 (px) | n |')
    lines.append('|---|---|---|')
    for arm in ('T', 'Tk', 'S', 'S240', 'F', 'M', 'S_cross', 'GEN'):
        vals = [r[6] for r in all_rows if r[1] == arm and not np.isnan(r[6])]
        if vals:
            lines.append(f'| {arm} | {np.median(vals):.2f} | {len(vals)} |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='XGaze 多相机 vs 单相机人脸建模对比')
    parser.add_argument('-j', '--jobs', type=int, default=12)
    parser.add_argument('-sb', '--subject_begin', type=int, default=0)
    parser.add_argument('-se', '--subject_end', type=int, default=80)
    args = parser.parse_args()

    sids = sorted(int(p.stem.replace('subject', '')) for p in Path(LM_DIR).glob('subject*.h5'))
    sids = sids[args.subject_begin:args.subject_end]
    log.info(f'对比实验: {len(sids)} 被试（subject{sids[0]:04d}..subject{sids[-1]:04d}）, '
             f'{args.jobs} 并行')
    t0 = time.time()
    all_rows, all_pairs, n_sels = [], [], []
    with Pool(args.jobs, initializer=init_worker) as pool:
        for i, res in enumerate(pool.imap_unordered(process_subject, sids), 1):
            if res['n_sel'] >= 2:
                all_rows.extend(res['rows'])
                all_pairs.extend(res['pairs'])
                n_sels.append(res['n_sel'])
            log.info(f'[{i}/{len(sids)}] '
                     f'{res["subject"]}: 选定 {res["n_sel"]} 相机, {res["sec"]:.0f}s')
    log.info(f'全部完成 {time.time() - t0:.0f}s, 选定相机数中位 {np.median(n_sels):.0f}')

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / 'per_subject.csv', 'w') as f:
        f.write('subject,arm,cam,n_train,n_kept,train_rms_px,test_rms_med_px,iod_mm\n')
        for r in all_rows:
            f.write(','.join(('nan' if (isinstance(x, float) and np.isnan(x))
                              else f'{x:.3f}' if isinstance(x, float) else str(x))
                             for x in r) + '\n')
    with open(METRICS_DIR / 'pairwise_models_mm.csv', 'w') as f:
        f.write('subject,cam_a,cam_b,kabsch_rms_mm\n')
        for p in all_pairs:
            f.write(f'{p[0]},{p[1]},{p[2]},{p[3]:.3f}\n')
    (METRICS_DIR / 'aggregate.md').write_text(aggregate(all_rows) + '\n')
    mm = [p[3] for p in all_pairs]
    log.info(f'模型两两结构差异: 中位 {np.median(mm):.2f} mm, p90 {np.percentile(mm, 90):.2f} '
             f'mm, 最大 {max(mm):.2f} mm (n={len(mm)} 对)')
    log.info(f'明细 {METRICS_DIR / "per_subject.csv"}\n{aggregate(all_rows)}')


if __name__ == '__main__':
    main()
