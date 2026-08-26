"""逐人个性化人脸建模共享核心（方案一：单相机多帧联合 BA，逐相机组交付）

自 xgaze/personalized_face_model.py（已全量验证）提炼，算法行为一致；差异：
- 像素↔归一化坐标换算按相机 fx 参数化（xgaze 原版硬编码 13200 ≈ 其相机 fx 尺度）
- 刚性点数参数化（默认同一套 RIGID 28 点 / IDX6 6 点常量）

各数据集端口只负责：观测加载、按「相机组」切分（xgaze/eve=相机、gazecapture=朝向、
mpii=唯一相机）、相机组选择、输出路径与命名；几何与优化全部走本模块。
共享约定：观测与模型都在归一化像坐标（undistortPoints 输出）中；RMS 以 px 报告
（乘该组 fx）。
"""
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

IDX6 = [35, 39, 89, 93, 78, 84]                     # 四眼角 + 两鼻底点
# 28 点刚性核心: 眼周(33-42, 87-96) + 鼻部(72-86); 剔除不稳点(眼球中心 34/38/88/92、鼻尖 86、94/95)
EXCLUDE = (34, 38, 86, 88, 92, 94, 95)
RIGID = [i for i in list(range(33, 43)) + list(range(72, 87)) + list(range(87, 97))
         if i not in EXCLUDE]
NP = len(RIGID)

# ===== 与 xgaze 验证一致的配置 =====
KEEP_FRAC = 0.5              # 好帧保留比例(按固定姿态三角化残差排序)
VIEW_ANGLE_MAX = 60.0        # 视图过滤: PnP 初始头部姿态角 < 60°
CAM_ANGLE_MAX = 40.0         # 相机组选择: 中位姿态角 < 40°
MIN_GROUP_VIEWS = 15         # 相机组可用训练视图下限
F_SCALE_PX = 5.0             # soft_l1 鲁棒损失尺度(px)
MAX_NFEV = 120


def f_px_of(K):
    """像素↔归一化换算尺度：fx/fy 均值"""
    return 0.5 * (K[0, 0] + K[1, 1])


def init_view(lm_px, K, dist, gen6):
    """单帧观测初始化：去畸变刚性核心 + GEN6 PnP 姿态。

    Returns:
        (lm_n (28,2) 归一化坐标, pose6 (rv,tv 拼接), 头部姿态角 deg)
    """
    lm_n = cv2.undistortPoints(lm_px.reshape(-1, 1, 2), K, dist).reshape(-1, 2)[RIGID]
    ret, rv, tv = cv2.solvePnP(gen6, lm_px[IDX6], K, dist, flags=cv2.SOLVEPNP_EPNP)
    ret, rv, tv = cv2.solvePnP(gen6, lm_px[IDX6], K, dist, rv, tv, True,
                               flags=cv2.SOLVEPNP_ITERATIVE)
    return lm_n, np.concatenate([rv.ravel(), tv.ravel()]), \
        float(np.degrees(np.linalg.norm(rv)))


def proj(X, rvs, tvs):
    """向量化投影: 模型 X (P,3) + 每视图姿态 -> 归一化像坐标 (V,P,2)."""
    R = Rotation.from_rotvec(rvs).as_matrix()
    x = np.einsum('fij,pj->fpi', R, X) + tvs[:, None, :]
    return x[..., :2] / x[..., 2:3]


def triangulate(lm_n, pv, n_points=NP):
    """固定姿态多帧 DLT 三角化(归一化坐标)."""
    R = Rotation.from_rotvec(pv[:, :3]).as_matrix()
    A = []
    for k in range(len(pv)):
        Pn = np.c_[R[k], pv[k, 3:6].reshape(3, 1)]
        A.append(np.stack([lm_n[k, :, 0][:, None] * Pn[2][None, :] - Pn[0][None, :],
                           lm_n[k, :, 1][:, None] * Pn[2][None, :] - Pn[1][None, :]]))
    A = np.concatenate(A)
    X = np.zeros((n_points, 3))
    for j in range(n_points):
        _, _, Vt = np.linalg.svd(A[:, j, :])
        X[j] = Vt[-1][:3] / Vt[-1][3]
    return X


def ba_shared(lm_n, pv, X0, f_px, n_points=NP, max_nfev=MAX_NFEV):
    """单相机联合 BA: 结构 + 逐帧姿态. 返回 (模型, 精化姿态, 逐帧 RMS px)."""
    NV = len(pv)

    def residuals(p):
        return (proj(p[:3 * n_points].reshape(n_points, 3),
                     p[3 * n_points:3 * n_points + 3 * NV].reshape(NV, 3),
                     p[3 * n_points + 3 * NV:].reshape(NV, 3)) - lm_n).ravel()

    x0 = np.concatenate([X0.ravel(), pv[:, :3].ravel(), pv[:, 3:].ravel()])
    S = lil_matrix((NV * n_points * 2, len(x0)), dtype=int)
    for v in range(NV):
        for j in range(n_points):
            r0 = (v * n_points + j) * 2
            S[r0:r0 + 2, 3 * j:3 * j + 3] = 1
            S[r0:r0 + 2, 3 * n_points + 3 * v:3 * n_points + 3 * v + 6] = 1
    r = least_squares(residuals, x0, jac_sparsity=S, loss='soft_l1',
                      f_scale=F_SCALE_PX / f_px, method='trf',
                      xtol=1e-12, ftol=1e-12, max_nfev=max_nfev)
    X = r.x[:3 * n_points].reshape(n_points, 3)
    pf = r.x[3 * n_points:].reshape(NV, 6)
    fr_rms = np.sqrt(np.mean(residuals(r.x).reshape(NV, n_points, 2) ** 2,
                             axis=(1, 2))) * f_px
    return X, pf, fr_rms


def kabsch(A, B):
    """求 R,t 使 R@A+t 逼近 B."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cb - R @ ca


def pnp_model(model, lm_n, f_px):
    """用模型对新视图做 PnP(归一化坐标), 返回 (rvec, tvec, 重投影 RMS px)."""
    ret, rv, tv = cv2.solvePnP(model, lm_n, np.eye(3), None, flags=cv2.SOLVEPNP_EPNP)
    ret, rv, tv = cv2.solvePnP(model, lm_n, np.eye(3), None, rv, tv, True,
                               flags=cv2.SOLVEPNP_ITERATIVE)
    R = cv2.Rodrigues(rv)[0]
    x = model @ R.T + tv.ravel()
    pr = x[:, :2] / x[:, 2:3]
    rms = float(np.sqrt(np.mean(np.sum(((pr - lm_n) * f_px) ** 2, axis=1))))
    return rv.ravel(), tv.ravel(), rms


def model_group(train_views, f_px, n_points=NP, keep_frac=KEEP_FRAC):
    """建模一个相机组：train 视图 -> 三角化 -> 好帧保留 -> 联合 BA。

    Args:
        train_views: [(lm_n (P,2), pose6)] 已通过视角过滤的训练视图
    Returns:
        dict(model=X, train_rms(中位 px), n_kept, n_train)
    """
    lm_t = np.stack([v[0] for v in train_views])
    pv_t = np.stack([v[1] for v in train_views])
    X0 = triangulate(lm_t, pv_t, n_points)
    pr = proj(X0, pv_t[:, :3], pv_t[:, 3:])
    fr_rms = np.sqrt(np.mean(((pr - lm_t) * f_px) ** 2, axis=(1, 2)))
    keep = np.argsort(fr_rms)[:max(4, int(len(train_views) * keep_frac))]
    X, pf, ba_rms = ba_shared(lm_t[keep], pv_t[keep], X0, f_px, n_points)
    return {'model': X, 'train_rms': float(np.median(ba_rms)),
            'n_kept': int(len(keep)), 'n_train': len(train_views)}


def eval_group(model, test_views, f_px, gen6=None, idx6_rows=None):
    """留出视图诊断：个性化模型 PnP RMS +（可选）通用模型基线，中位 px。"""
    errs, errs_gen = [], []
    for lm_n, _ in test_views:
        _, _, rms = pnp_model(model, lm_n, f_px)
        errs.append(rms)
        if gen6 is not None:
            _, _, rms_g = pnp_model(gen6, lm_n[idx6_rows], f_px)
            errs_gen.append(rms_g)
    return (float(np.median(errs)) if errs else float('nan'),
            float(np.median(errs_gen)) if errs_gen else float('nan'))


def canonical_of(models):
    """Kabsch 融合参考模型（对齐第一个组，逐元素中位）。"""
    names = list(models)
    a = names[0]
    stack = []
    for g in names:
        if g == a:
            stack.append(models[g])
            continue
        R, t = kabsch(models[g], models[a])
        stack.append((R @ models[g].T).T + t)
    return np.median(np.stack(stack), axis=0)


def save_models(sub_dir, models, canonical, idx6_rows, summary_lines, overwrite=False):
    """按 xgaze 输出约定落盘一个被试/session 的全部组模型与 summary。

    Args:
        models: {组名: (28,3) 模型}，组名形如 'cam00' / 'ori1'——
                写 {组名}_model6.txt / {组名}_model28.txt（6 点为 28 点的 IDX6 行子集）
        canonical: Kabsch 融合参考模型
        summary_lines: 每组一行「组名 train_rms test_rms iod」
    """
    import os
    marker = os.path.join(sub_dir, 'summary.txt')
    if os.path.exists(marker) and not overwrite:
        return False
    os.makedirs(sub_dir, exist_ok=True)
    for name, model in models.items():
        np.savetxt(os.path.join(sub_dir, '{}_model6.txt'.format(name)),
                   model[idx6_rows], fmt='%.6f')
        np.savetxt(os.path.join(sub_dir, '{}_model28.txt'.format(name)),
                   model, fmt='%.6f')
    np.savetxt(os.path.join(sub_dir, 'canonical_model28.txt'), canonical, fmt='%.6f')
    with open(marker, 'w') as f:
        f.write('# 刚性 28 点在 insightface 106 点中的索引: {}\n'.format(RIGID))
        f.write('# 6 点模型对应的 106 点索引(IDX6): {}\n'.format(IDX6))
        f.write('# group  train_rms_px  test_rms_px  iod_mm\n')
        f.writelines(summary_lines)
    return True
