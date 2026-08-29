"""EVE 多相机 vs 单相机人脸 3D 模型重建对比实验（2026-08-28）

问题：同一被试的人脸刚性 28 点模型，
  - 单相机（现行平台方案）：每相机独立 BA（仅本相机 60 训练视图）→ 逐相机交付 cam{cc}_model
  - 多相机：4 相机训练视图合并成一个联合 BA（每视图独立姿态、逐相机 f_px 损失尺度）
    → 每被试只交付一个模型
评判标准（用户指定）：建好的 3D 模型投影到像素平面与对应 2d 关键点的误差（PnP 拟合后
的重投影 RMS，px）。主指标 = 各相机**留出测试帧**上的中位 RMS。

实验臂（同一被试、同一套观测采样）：
  S     单相机逐组（现行方案：三角化 → 剔坏帧 → BA）
  T     消融：仅三角化（全部训练帧，不剔帧不 BA）
  Tk    消融：剔除坏帧后仅三角化（不 BA）——T→Tk→S 拆出剔帧与 BA 的各自贡献
  S240  单相机 basler 视图数对照（240 训练视图）——区分「相机多样性」与「帧数多」
  F     模型级融合（各相机 S 模型 Kabsch 对齐取逐点中位，core.canonical_of）
  M     多相机联合（4 相机 × 60 = ~240 训练视图观测级合并 BA，逐视图 f_px）
  GEN   通用 50 点模型基线（context）
附加：S(basler) 模型跨相机评估（单相机模型对未见相机的泛化）；
      M 在未通过单相机选择准则（中位角 ≥40°）相机上的表现。

防泄漏协议：每相机统一测试集 = linspace(0,N-1,120) 的奇数位 60 帧；所有实验臂的训练集
必须与统一测试集索引距离 >2 帧（5Hz 下相邻帧近重复）。S/M 天然满足（偶数位采样），
S240 从 linspace(0,N-1,480) 偶数位中剔除测试集邻域后取 240。

输入: landmarks 索引 h5 + 原始 <subject>/<step>/<cam>.h5 内参（mp4 已去畸变 → 畸变零）
输出: metrics/multi_camera/{per_subject.csv, aggregate.md 片段, run.log}
用法（仓库根目录运行；CPU 即可）:
  .../multi_camera_study.py [-j 12] [-sb 0 -se 44]
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')   # 多进程并行前提：单线程 BLAS
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys
import json
import time
import argparse
from pathlib import Path
from multiprocessing import Pool

import cv2
import h5py
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))          # get_face_model/
sys.path.insert(0, str(PROJECT_ROOT))

import face_model_core as core                                        # noqa: E402
from utils.logger import get_logger                                   # noqa: E402

log = get_logger('preprocess.specific_face_model.eve.multicam_study')

LM_ROOT = '/media/yanglinxuan/zyx/EVE_dataset/eve_dataset/landmarks'
RAW_DIR = '/media/yanglinxuan/zyx/EVE_dataset/eve_dataset'
METRICS_DIR = Path(__file__).resolve().parent / 'metrics' / 'multi_camera'
FACE_MODEL_FILE = Path(__file__).resolve().parents[3] / 'zhang2015-insightface' / 'face_model_xgaze.txt'
GEN_ROWS = [20, 23, 26, 29, 15, 19]

N_SAMPLE = 120          # 每相机采样帧数（与现行建模协议一致）
N_SAMPLE_240 = 480      # S240 臂的 basler 采样帧数
LEAK_GUARD = 2          # 训练集与统一测试集的最小索引距离


# ---------------------------------------------------------------- 观测准备
def load_K(subject, cam_name, steps):
    for step in steps:
        p = Path(RAW_DIR) / subject / step / (cam_name + '.h5')
        if p.is_file():
            with h5py.File(p, 'r') as f:
                return np.array(f['camera_matrix'], dtype=float)
    raise FileNotFoundError(f'{RAW_DIR}/{subject} 下未找到 {cam_name}.h5')


def build_views(lm_all, rows, K, gen6):
    """与现行建模协议一致：undistort 刚性 28 点 + GEN6 PnP 初始姿态，过滤姿态角 <60°"""
    dist = np.zeros((1, 5), dtype=float)
    vs, angles = [], []
    for r in rows:
        lm_n, pose, ang = core.init_view(lm_all[r].astype(np.float64), K, dist, gen6)
        if ang < core.VIEW_ANGLE_MAX:
            vs.append((lm_n, pose))
            angles.append(ang)
    return vs, angles


# ---------------------------------------------------------------- 多相机 BA
def ba_multicam(lm_t, pv_t, X0, f_pxs, n_points=core.NP):
    """多相机联合 BA：与 core.ba_shared 同构，差异 = 残差按逐视图 f_px 换算到像素单位
    （soft_l1 鲁棒损失在像素域统一 f_scale=5px，等价于逐相机各自的归一化域 f_scale）。"""
    NV = len(pv_t)

    def residuals(p):
        X = p[:3 * n_points].reshape(n_points, 3)
        rv = p[3 * n_points:3 * n_points + 3 * NV].reshape(NV, 3)
        tv = p[3 * n_points + 3 * NV:].reshape(NV, 3)
        pr = core.proj(X, rv, tv)
        return ((pr - lm_t) * f_pxs[:, None, None]).ravel()

    x0 = np.concatenate([X0.ravel(), pv_t[:, :3].ravel(), pv_t[:, 3:].ravel()])
    S = lil_matrix((NV * n_points * 2, len(x0)), dtype=int)
    for v in range(NV):
        for j in range(n_points):
            r0 = (v * n_points + j) * 2
            S[r0:r0 + 2, 3 * j:3 * j + 3] = 1
            S[r0:r0 + 2, 3 * n_points + 3 * v:3 * n_points + 3 * v + 6] = 1
    r = least_squares(residuals, x0, jac_sparsity=S, loss='soft_l1',
                      f_scale=core.F_SCALE_PX, method='trf',
                      xtol=1e-12, ftol=1e-12, max_nfev=core.MAX_NFEV)
    X = r.x[:3 * n_points].reshape(n_points, 3)
    pf = r.x[3 * n_points:].reshape(NV, 6)
    fr_rms = np.sqrt(np.mean(residuals(r.x).reshape(NV, n_points, 2) ** 2,
                             axis=(1, 2)))
    return X, pf, fr_rms


def model_group_multicam(train_views, f_pxs):
    """多相机版 model_group：固定姿态三角化 → 好帧保留 → 联合 BA（逐视图 f_px）"""
    lm_t = np.stack([v[0] for v in train_views])
    pv_t = np.stack([v[1] for v in train_views])
    X0 = core.triangulate(lm_t, pv_t)
    pr = core.proj(X0, pv_t[:, :3], pv_t[:, 3:])
    fr_rms = np.sqrt(np.mean(((pr - lm_t) * f_pxs[:, None, None]) ** 2, axis=(1, 2)))
    keep = np.argsort(fr_rms)[:max(4, int(len(train_views) * core.KEEP_FRAC))]
    X, pf, ba_rms = ba_multicam(lm_t[keep], pv_t[keep], X0, f_pxs[keep])
    return {'model': X, 'train_rms': float(np.median(ba_rms)),
            'n_kept': int(len(keep)), 'n_train': len(train_views)}


def eval_views(model, views, f_px, idx6_rows=None, gen6=None):
    """留出视图评估：个性化模型 28 点 PnP RMS（+可选通用 6 点基线），中位 px"""
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
def process_subject(args):
    split, subject, gen6 = args
    t0 = time.time()
    idx6_rows = [core.RIGID.index(i) for i in core.IDX6]
    lm_path = Path(LM_ROOT) / split / f'{subject}.h5'
    with h5py.File(lm_path, 'r') as f:
        cam_all = f['cam_index'][:].ravel()
        lm_all = f['facial_landmarks_2d'][:]
        cameras = json.loads(f.attrs['cameras'])
        steps = json.loads(f.attrs['steps'])

    rows_out = []          # (subject, arm, cam, n_train, n_kept, train_rms, test_rms, iod)
    per_cam = {}           # cam -> {'test_views', 'f_px', 'eligible', 'med_angle'}

    # ---- 逐相机观测准备 + 统一测试集 ----
    for c, cam_name in enumerate(cameras):
        K = load_K(subject, cam_name, steps)
        f_px = core.f_px_of(K)
        rows_c = np.where(cam_all == c)[0]
        if len(rows_c) < 20:
            continue
        n_sample = min(N_SAMPLE, len(rows_c))
        sel = rows_c[np.linspace(0, len(rows_c) - 1, n_sample).astype(int)]
        train_rows, test_rows = sel[0::2], sel[1::2]

        train_views, angles = build_views(lm_all, train_rows, K, gen6)
        test_views, _ = build_views(lm_all, test_rows, K, gen6)
        med = float(np.median(angles)) if angles else 180.0
        eligible = (med < core.CAM_ANGLE_MAX and len(train_views) >= core.MIN_GROUP_VIEWS
                    and bool(test_views))
        per_cam[c] = {'name': cam_name, 'K': K, 'f_px': f_px, 'rows_c': rows_c,
                      'train_views': train_views, 'test_views': test_views,
                      'eligible': eligible, 'med_angle': med}

    # ---- 臂 S：单相机逐组（现行方案）+ 消融臂 T/Tk ----
    s_models = {}
    for c, d in per_cam.items():
        if not d['eligible']:
            continue
        lm_t = np.stack([v[0] for v in d['train_views']])
        pv_t = np.stack([v[1] for v in d['train_views']])
        fpx = d['f_px']
        # T：仅三角化（全部训练帧；train 误差 = 初始姿态回投影残差）
        X0 = core.triangulate(lm_t, pv_t)
        pr = core.proj(X0, pv_t[:, :3], pv_t[:, 3:])
        fr_rms = np.sqrt(np.mean(((pr - lm_t) * fpx) ** 2, axis=(1, 2)))
        te0, _ = eval_views(X0, d['test_views'], fpx)
        rows_out.append((subject, 'T', d['name'], len(lm_t), len(lm_t),
                         float(np.median(fr_rms)), te0, iod_of(X0)))
        # Tk：剔除坏帧后仅三角化（好帧判定与 core.model_group 完全一致）
        keep = np.argsort(fr_rms)[:max(4, int(len(lm_t) * core.KEEP_FRAC))]
        X0k = core.triangulate(lm_t[keep], pv_t[keep])
        tek, _ = eval_views(X0k, d['test_views'], fpx)
        rows_out.append((subject, 'Tk', d['name'], len(keep), len(keep),
                         float(np.median(fr_rms[keep])), tek, iod_of(X0k)))
        # S：完整管线
        res = core.model_group(d['train_views'], d['f_px'])
        s_models[c] = res
        te, _ = eval_views(res['model'], d['test_views'], d['f_px'])
        rows_out.append((subject, 'S', d['name'], res['n_train'], res['n_kept'],
                         res['train_rms'], te, iod_of(res['model'])))

    # ---- 臂 S240：单相机 basler 视图数对照 ----
    c0 = 0
    if c0 in per_cam and per_cam[c0]['eligible']:
        d = per_cam[c0]
        rows_c = d['rows_c']
        test_idx = set(rows_c[np.linspace(0, len(rows_c) - 1,
                                          min(N_SAMPLE, len(rows_c))).astype(int)][1::2].tolist())
        n_s = min(N_SAMPLE_240, len(rows_c))
        sel240 = rows_c[np.linspace(0, len(rows_c) - 1, n_s).astype(int)][0::2]
        guard = np.array(sorted(test_idx))
        keep = np.array([abs(int(r) - guard).min() > LEAK_GUARD for r in sel240])
        sel240 = sel240[keep][:240]
        if len(sel240) >= core.MIN_GROUP_VIEWS:
            train240, _ = build_views(lm_all, sel240, d['K'], gen6)
            res = core.model_group(train240, d['f_px'])
            te, _ = eval_views(res['model'], d['test_views'], d['f_px'])
            rows_out.append((subject, 'S240', d['name'], res['n_train'], res['n_kept'],
                             res['train_rms'], te, iod_of(res['model'])))

    # ---- 臂 F：模型级融合（各相机 S 模型 Kabsch 对齐取逐点中位，即 core.canonical_of）----
    if len(s_models) >= 2:
        fused = core.canonical_of({c: r['model'] for c, r in s_models.items()})
        for c, d in per_cam.items():
            if not d['test_views']:
                continue
            te, _ = eval_views(fused, d['test_views'], d['f_px'])
            rows_out.append((subject, 'F', d['name'], sum(r['n_train'] for r in s_models.values()),
                             0, float('nan'), te, iod_of(fused)))

    # ---- 臂 M：多相机联合（全部 <60° 视图并入，含未通过单相机选择准则的相机） ----
    pooled, pooled_fpx, pooled_cam = [], [], []
    for c, d in per_cam.items():
        for v in d['train_views']:
            pooled.append(v)
            pooled_fpx.append(d['f_px'])
            pooled_cam.append(c)
    if len(pooled) >= core.MIN_GROUP_VIEWS * 2:
        res = model_group_multicam(pooled, np.array(pooled_fpx))
        for c, d in per_cam.items():
            if not d['test_views']:
                continue
            te, _ = eval_views(res['model'], d['test_views'], d['f_px'])
            rows_out.append((subject, 'M', d['name'], res['n_train'], res['n_kept'],
                             res['train_rms'], te, iod_of(res['model'])))
        # GEN 基线 + S(basler) 跨相机（附带记录，每相机一次）
        for c, d in per_cam.items():
            if not d['test_views']:
                continue
            _, te_gen = eval_views(res['model'], d['test_views'], d['f_px'],
                                   idx6_rows, gen6)
            rows_out.append((subject, 'GEN', d['name'], 0, 0, float('nan'),
                             te_gen, float('nan')))
        if 0 in s_models:
            for c, d in per_cam.items():
                if c == 0 or not d['test_views']:
                    continue
                te, _ = eval_views(s_models[0]['model'], d['test_views'], d['f_px'])
                rows_out.append((subject, 'S_cross', d['name'], 0, 0, float('nan'),
                                 te, float('nan')))

    return {'subject': subject, 'n_cams': len(per_cam),
            'n_eligible': sum(d['eligible'] for d in per_cam.values()),
            'rows': rows_out, 'sec': time.time() - t0}


# ---------------------------------------------------------------- 汇总
def aggregate(all_rows):
    """(arm, cam) -> 中位 test_rms / train_rms / iod；跨被试取中位"""
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
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='EVE 多相机 vs 单相机人脸建模对比')
    parser.add_argument('-j', '--jobs', type=int, default=12)
    parser.add_argument('-sb', '--subject_begin', type=int)
    parser.add_argument('-se', '--subject_end', type=int)
    args = parser.parse_args()

    gen6 = np.loadtxt(FACE_MODEL_FILE)[GEN_ROWS, :]
    subjects = [(sp, p.stem) for sp in ('train', 'test')
                for p in sorted(Path(LM_ROOT, sp).glob('*.h5'))]
    sb = args.subject_begin if args.subject_begin is not None else 0
    se = args.subject_end if args.subject_end is not None else len(subjects)
    subjects = subjects[sb:se]
    log.info(f'对比实验: {len(subjects)} 被试, {args.jobs} 并行')

    t0 = time.time()
    all_rows, n_elig = [], []
    with Pool(args.jobs) as pool:
        for i, res in enumerate(pool.imap_unordered(
                process_subject, [(sp, s, gen6) for sp, s in subjects]), 1):
            if res is None:
                continue
            all_rows.extend(res['rows'])
            n_elig.append(res['n_eligible'])
            log.info(f'[{i}/{len(subjects)}] {res["subject"]}: '
                     f'{res["n_cams"]} 相机 / {res["n_eligible"]} 单相机可用组, '
                     f'{res["sec"]:.0f}s')
    log.info(f'全部完成 {time.time() - t0:.0f}s, 单相机可用组中位 {np.median(n_elig):.0f}')

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = METRICS_DIR / 'per_subject.csv'
    with open(csv_path, 'w') as f:
        f.write('subject,arm,cam,n_train,n_kept,train_rms_px,test_rms_med_px,iod_mm\n')
        for r in all_rows:
            f.write(','.join(('nan' if (isinstance(x, float) and np.isnan(x))
                              else f'{x:.3f}' if isinstance(x, float) else str(x))
                             for x in r) + '\n')
    table = aggregate(all_rows)
    (METRICS_DIR / 'aggregate.md').write_text(table + '\n')
    log.info(f'明细 {csv_path}\n{table}')


if __name__ == '__main__':
    main()
