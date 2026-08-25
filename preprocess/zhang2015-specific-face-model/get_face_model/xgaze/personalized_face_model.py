"""ETH-XGaze 逐被试个性化人脸模型(方案一: 单相机多帧 BA, 逐相机交付)

输入:
  - insightface 106 关键点 h5(原图像素坐标): /home/hitsz/dataset/xgaze_insightface_224/subject*.h5
  - 相机内参 xml: cam_calibration/cam{cc}.xml
输出 /home/hitsz/dataset/xgaze_specific_face_model_224/face_models/subject{id:04d}/ 下(txt, np.loadtxt 直接读):
  - cam{cc}_model6.txt / cam{cc}_model28.txt: 视角正常相机的个性化模型(6 点=IDX6, 28 点=刚性核心)
  - canonical_model28.txt: Kabsch 融合参考模型
  - summary.txt: 相机列表 + 质量诊断
建模指标留档(管线目录 <本目录>/metrics/):
  - subject{id:04d}.txt: 逐相机 train/test RMS、通用模型基线 RMS(同留出帧对比)、IOD、留用帧数
  - summary_all.csv: 全部被试汇总(subject, n_cams, train/test/通用 中位, 改善倍数, IOD)
  注: 逐帧精化姿态(refined_poses)不在此保存, 归入后续预处理流水线
用法(在仓库根目录运行):
  ~/anaconda3/envs/opengaze/bin/python preprocess/zhang2015-specific-face-model/get_face_model/xgaze/personalized_face_model.py -sb 0 -se 120 --overwrite
"""
import os
import sys
import argparse
import time

import cv2
import h5py
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# utils 包在仓库根（xgaze → get_face_model → zhang2015-specific-face-model → preprocess → 仓库根）
sys.path.insert(0, os.path.abspath(os.path.join(PROJECT_ROOT, '..', '..', '..', '..')))

from utils.logger import get_logger  # noqa: E402

log = get_logger('preprocess.specific_face_model.xgaze')
LM_DIR = '/media/hitsz/ylx/xgaze_insightface_224'
CALIB_DIR = '/media/hitsz/Expansion/xgaze_raw/calibration/cam_calibration'
OUT_DIR = '/home/hitsz/dataset/xgaze_specific_face_model_224/face_models'
METRICS_DIR = os.path.join(PROJECT_ROOT, 'metrics')   # 建模指标留档(管线目录)
# 通用 50 点模型（与 zhang2015-insightface 管线共用同一份，内容与官方 face_model.txt 一致）
FACE_MODEL_FILE = os.path.join(PROJECT_ROOT, '..', '..', '..', 'zhang2015-insightface', 'face_model_xgaze.txt')

IDX6 = [35, 39, 89, 93, 78, 84]                     # 四眼角 + 两鼻底点
GEN_ROWS = [20, 23, 26, 29, 15, 19]                 # face_model.txt 中对应 6 点的行
# 28 点刚性核心: 眼周(33-42, 87-96) + 鼻部(72-86); 剔除不稳点(眼球中心 34/38/88/92、鼻尖 86、94/95)
EXCLUDE = (34, 38, 86, 88, 92, 94, 95)
RIGID = [i for i in list(range(33, 43)) + list(range(72, 87)) + list(range(87, 97))
         if i not in EXCLUDE]
NP = len(RIGID)

# ===== 验证确定的配置(subject0000 对比实验) =====
N_TRAIN, N_TEST = 60, 60     # 每相机采样帧数: 120 帧均匀采样, 奇偶分为 train/test
KEEP_FRAC = 0.5              # 好帧保留比例(按固定姿态三角化残差排序)
VIEW_ANGLE_MAX = 60.0        # 视图过滤: PnP 初始头部姿态角 < 60°(更极端的视角检测失真)
CAM_ANGLE_MAX = 40.0         # 相机选择: 中位姿态角 < 40°(视角正常、角度较小的相机)
MIN_CAM_VIEWS = 15           # 相机可用训练视图下限
MIN_CAMERAS = 6              # 被试至少需要的相机数
F_SCALE_PX = 5.0             # soft_l1 鲁棒损失尺度
MAX_NFEV = 120


def load_calibrations():
    Ks, dists = {}, {}
    for c in range(18):
        fs = cv2.FileStorage(os.path.join(CALIB_DIR, 'cam{:02d}.xml'.format(c)),
                             cv2.FILE_STORAGE_READ)
        Ks[c] = fs.getNode('Camera_Matrix').mat()
        dists[c] = fs.getNode('Distortion_Coefficients').mat()
        fs.release()
    return Ks, dists


def proj(X, rvs, tvs):
    """向量化投影: 模型 X (P,3) + 每视图姿态 -> 归一化像坐标 (V,P,2)."""
    R = Rotation.from_rotvec(rvs).as_matrix()
    x = np.einsum('fij,pj->fpi', R, X) + tvs[:, None, :]
    return x[..., :2] / x[..., 2:3]


def triangulate(lm_n, pv):
    """固定姿态多帧 DLT 三角化(归一化坐标)."""
    R = Rotation.from_rotvec(pv[:, :3]).as_matrix()
    A = []
    for k in range(len(pv)):
        Pn = np.c_[R[k], pv[k, 3:6].reshape(3, 1)]
        A.append(np.stack([lm_n[k, :, 0][:, None] * Pn[2][None, :] - Pn[0][None, :],
                           lm_n[k, :, 1][:, None] * Pn[2][None, :] - Pn[1][None, :]]))
    A = np.concatenate(A)
    X = np.zeros((NP, 3))
    for j in range(NP):
        _, _, Vt = np.linalg.svd(A[:, j, :])
        X[j] = Vt[-1][:3] / Vt[-1][3]
    return X


def ba_shared(lm_n, pv, X0):
    """单相机联合 BA: 结构 + 逐帧姿态. 返回 (模型, 精化姿态, 逐帧RMS px)."""
    NV = len(pv)

    def residuals(p):
        return (proj(p[:3 * NP].reshape(NP, 3), p[3 * NP:3 * NP + 3 * NV].reshape(NV, 3),
                     p[3 * NP + 3 * NV:].reshape(NV, 3)) - lm_n).ravel()

    x0 = np.concatenate([X0.ravel(), pv[:, :3].ravel(), pv[:, 3:].ravel()])
    S = lil_matrix((NV * NP * 2, len(x0)), dtype=int)
    for v in range(NV):
        for j in range(NP):
            r0 = (v * NP + j) * 2
            S[r0:r0 + 2, 3 * j:3 * j + 3] = 1
            S[r0:r0 + 2, 3 * NP + 3 * v:3 * NP + 3 * v + 6] = 1
    r = least_squares(residuals, x0, jac_sparsity=S, loss='soft_l1',
                      f_scale=F_SCALE_PX / 13200.0, method='trf',
                      xtol=1e-12, ftol=1e-12, max_nfev=MAX_NFEV)
    X = r.x[:3 * NP].reshape(NP, 3)
    pf = r.x[3 * NP:].reshape(NV, 6)
    fr_rms = np.sqrt(np.mean(residuals(r.x).reshape(NV, NP, 2) ** 2, axis=(1, 2))) * 13200.0
    return X, pf, fr_rms


def kabsch(A, B):
    """求 R,t 使 R@A+t 逼近 B."""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cb - R @ ca


def pnp_model(model, lm_n):
    """用模型对新视图做 PnP(归一化坐标), 返回 (rvec, tvec, 重投影RMS px)."""
    ret, rv, tv = cv2.solvePnP(model, lm_n, np.eye(3), None, flags=cv2.SOLVEPNP_EPNP)
    ret, rv, tv = cv2.solvePnP(model, lm_n, np.eye(3), None, rv, tv, True,
                               flags=cv2.SOLVEPNP_ITERATIVE)
    R = cv2.Rodrigues(rv)[0]
    x = model @ R.T + tv.ravel()
    pr = x[:, :2] / x[:, 2:3]
    rms = float(np.sqrt(np.mean(np.sum(((pr - lm_n) * 13200.0) ** 2, axis=1))))
    return rv.ravel(), tv.ravel(), rms


def process_subject(sid, Ks, dists, gen6):
    h5_path = os.path.join(LM_DIR, 'subject{:04d}.h5'.format(sid))
    if not os.path.isfile(h5_path):
        log.info('  h5 不存在, 跳过: {}'.format(h5_path))
        return None
    with h5py.File(h5_path, 'r') as f:
        fr_all = f['frame_index'][:].ravel()
        cam_all = f['cam_index'][:].ravel()
        lm_all = f['facial_landmarks_2d'][:]

    # 帧采样: 120 帧均匀采样 -> train/test 各 60(与验证协议一致)
    frames_universe = sorted(set(fr_all.tolist()))
    n_sample = min(N_TRAIN + N_TEST, len(frames_universe))
    sel = np.array(frames_universe)[np.linspace(0, len(frames_universe) - 1, n_sample).astype(int)]
    train_frames = sorted(sel[0::2].tolist())
    test_frames = sorted(sel[1::2].tolist())

    # 预处理所有 (相机, 采样帧) 视图: 去畸变刚性核心观测 + GEN6 PnP 姿态初值
    obs, poses_init, cam_angle = {}, {}, {}
    for c in range(18):
        m = (cam_all == c) & np.isin(fr_all, sel)
        angles = []
        for r_, fr_ in zip(np.where(m)[0], fr_all[m]):
            lm_px = lm_all[r_].astype(np.float64)
            lm_n = cv2.undistortPoints(lm_px.reshape(-1, 1, 2), Ks[c], dists[c]).reshape(-1, 2)[RIGID]
            ret, rv, tv = cv2.solvePnP(gen6, lm_px[IDX6], Ks[c], dists[c], flags=cv2.SOLVEPNP_EPNP)
            ret, rv, tv = cv2.solvePnP(gen6, lm_px[IDX6], Ks[c], dists[c], rv, tv, True,
                                       flags=cv2.SOLVEPNP_ITERATIVE)
            obs[(c, int(fr_))] = lm_n
            poses_init[(c, int(fr_))] = np.concatenate([rv.ravel(), tv.ravel()])
            angles.append(np.degrees(np.linalg.norm(rv)))
        cam_angle[c] = float(np.median(angles)) if angles else 180.0

    # 相机选择: 视角正常(中位姿态角小)且训练视图充足
    ok_view = lambda vk: np.degrees(np.linalg.norm(poses_init[vk][:3])) < VIEW_ANGLE_MAX
    sel_cams = []
    for c in range(18):
        n_views = sum(1 for fr in train_frames if (c, fr) in obs and ok_view((c, fr)))
        if cam_angle[c] < CAM_ANGLE_MAX and n_views >= MIN_CAM_VIEWS:
            sel_cams.append(c)
    log.info('  相机选择(中位角<{:.0f}°): {} | 各相机中位角: {}'.format(
        CAM_ANGLE_MAX, sel_cams,
        ' '.join('{:.0f}'.format(cam_angle[c]) for c in range(18))))
    if len(sel_cams) < MIN_CAMERAS:
        log.warning('  仅 {} 台相机 < {}（仍建模，标记偏少）'.format(
            len(sel_cams), MIN_CAMERAS))
    if not sel_cams:
        return None

    models, train_rms, test_rms, test_gen, iods, n_kept = {}, {}, {}, {}, {}, {}
    idx6_rows = [RIGID.index(i) for i in IDX6]
    for c in sel_cams:
        # ---- 建模: train 帧, 三角化 -> 好帧 50% -> BA ----
        vks = [vk for vk in ((c, fr) for fr in train_frames) if vk in obs and ok_view(vk)]
        lm_t = np.stack([obs[vk] for vk in vks])
        pv_t = np.stack([poses_init[vk] for vk in vks])
        X0 = triangulate(lm_t, pv_t)
        pr = proj(X0, pv_t[:, :3], pv_t[:, 3:])
        fr_rms = np.sqrt(np.mean(((pr - lm_t) * 13200.0) ** 2, axis=(1, 2)))
        keep = np.argsort(fr_rms)[:max(4, int(len(vks) * KEEP_FRAC))]
        Xc, pf, ba_rms = ba_shared(lm_t[keep], pv_t[keep], X0)
        models[c] = Xc
        train_rms[c] = float(np.median(ba_rms))
        n_kept[c] = int(len(keep))

        # ---- 留出帧诊断: 个性化模型 + 通用模型基线(同帧同观测对比) ----
        errs, errs_gen = [], []
        for fr in test_frames:
            vk = (c, fr)
            if vk not in obs or not ok_view(vk):
                continue
            _, _, rms = pnp_model(Xc, obs[vk])
            errs.append(rms)
            _, _, rms_g = pnp_model(gen6, obs[vk][idx6_rows])
            errs_gen.append(rms_g)
        test_rms[c] = float(np.median(errs)) if errs else float('nan')
        test_gen[c] = float(np.median(errs_gen)) if errs_gen else float('nan')
        iods[c] = float(np.linalg.norm(Xc[RIGID.index(35)] - Xc[RIGID.index(93)]))

    # ---- 融合参考模型(Kabsch 对齐第一台选定相机, 中位数) ----
    a = sel_cams[0]
    stack = []
    for c in sel_cams:
        if c == a:
            stack.append(models[c])
            continue
        R, t = kabsch(models[c], models[a])
        stack.append((R @ models[c].T).T + t)
    canonical = np.median(np.stack(stack), axis=0)

    # ---- 保存(txt, 与官方 face_model.txt 同构, np.loadtxt 直接读取) ----
    sub_dir = os.path.join(OUT_DIR, 'subject{:04d}'.format(sid))
    os.makedirs(sub_dir, exist_ok=True)
    for c in sel_cams:
        np.savetxt(os.path.join(sub_dir, 'cam{:02d}_model6.txt'.format(c)),
                   models[c][[RIGID.index(i) for i in IDX6]], fmt='%.6f')
        np.savetxt(os.path.join(sub_dir, 'cam{:02d}_model28.txt'.format(c)),
                   models[c], fmt='%.6f')
    np.savetxt(os.path.join(sub_dir, 'canonical_model28.txt'), canonical, fmt='%.6f')
    with open(os.path.join(sub_dir, 'summary.txt'), 'w') as f:
        f.write('# 刚性 28 点在 insightface 106 点中的索引: {}\n'.format(RIGID))
        f.write('# 6 点模型对应的 106 点索引(IDX6): {}\n'.format(IDX6))
        f.write('# camera  train_rms_px  test_rms_px  iod_mm\n')
        for c in sel_cams:
            f.write('cam{:02d}  {:.2f}  {:.2f}  {:.1f}\n'.format(
                c, train_rms[c], test_rms[c], iods[c]))

    # ---- 建模指标留档(管线目录 metrics/): 含通用模型基线对比 ----
    med_tr = float(np.median(list(train_rms.values())))
    med_te = float(np.nanmedian(list(test_rms.values())))
    med_ge = float(np.nanmedian(list(test_gen.values())))
    mean_iod = float(np.mean(list(iods.values())))
    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(os.path.join(METRICS_DIR, 'subject{:04d}.txt'.format(sid)), 'w') as f:
        f.write('# 相机数 {} | train RMS 中位 {:.2f} px | test RMS 中位 {:.2f} px | '
                '通用模型 test RMS 中位 {:.2f} px | 改善 {:.1f}x | IOD 均值 {:.1f} mm\n'.format(
                    len(sel_cams), med_tr, med_te, med_ge,
                    med_ge / max(med_te, 1e-6), mean_iod))
        f.write('# camera  n_train_kept  train_rms_px  test_rms_px  '
                'test_rms_generic_px  iod_mm\n')
        for c in sel_cams:
            f.write('cam{:02d}  {}  {:.2f}  {:.2f}  {:.2f}  {:.1f}\n'.format(
                c, n_kept[c], train_rms[c], test_rms[c], test_gen[c], iods[c]))
    log.info('  建模 {} 相机 | train RMS 中位 {:.2f} px | test RMS 中位 {:.2f} px | '
             '通用基线 {:.2f} px ({:.1f}x) | IOD {:.1f} mm -> {}'.format(
                 len(sel_cams), med_tr, med_te, med_ge,
                 med_ge / max(med_te, 1e-6), mean_iod, sub_dir))
    return {'subject': 'subject{:04d}'.format(sid), 'n_cams': len(sel_cams),
            'train_med': med_tr, 'test_med': med_te, 'test_gen_med': med_ge,
            'imp': med_ge / max(med_te, 1e-6), 'iod_mean': mean_iod}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ETH-XGaze 逐人个性化人脸建模(方案一)')
    parser.add_argument('-sb', '--subject_begin', type=int, help='起始被试编号(含)')
    parser.add_argument('-se', '--subject_end', type=int, help='结束被试编号(不含)')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已存在的输出')
    args = parser.parse_args()

    gen6_full = np.loadtxt(FACE_MODEL_FILE)
    gen6 = gen6_full[GEN_ROWS, :]

    subject_begin = args.subject_begin if args.subject_begin is not None else 0
    subject_end = args.subject_end if args.subject_end is not None else subject_begin + 1

    Ks, dists = load_calibrations()
    rows = []
    for sid in range(subject_begin, subject_end):
        done_marker = os.path.join(OUT_DIR, 'subject{:04d}'.format(sid), 'summary.txt')
        if os.path.exists(done_marker) and not args.overwrite:
            log.info('subject{:04d}: 已存在, 跳过'.format(sid))
            continue
        t0 = time.time()
        log.info('subject{:04d}:'.format(sid))
        row = process_subject(sid, Ks, dists, gen6)
        if row is not None:
            rows.append(row)
        log.info('  用时 {:.1f}s'.format(time.time() - t0))

    # 汇总指标 CSV（每人一行）
    os.makedirs(METRICS_DIR, exist_ok=True)
    csv_path = os.path.join(METRICS_DIR, 'summary_all.csv')
    with open(csv_path, 'w') as f:
        f.write('subject,n_cams,train_rms_med_px,test_rms_med_px,'
                'test_rms_generic_med_px,improvement_x,iod_mean_mm\n')
        for r in rows:
            f.write('{subject},{n_cams},{train_med:.2f},{test_med:.2f},'
                    '{test_gen_med:.2f},{imp:.1f},{iod_mean:.1f}\n'.format(**r))
    if rows:
        log.info('完成: {} 个被试 | train 中位 {:.2f} px | test 中位 {:.2f} px | '
                 '通用基线中位 {:.2f} px | 改善中位 {:.1f}x | 指标留档 {}'.format(
                     len(rows),
                     np.median([r['train_med'] for r in rows]),
                     np.median([r['test_med'] for r in rows]),
                     np.median([r['test_gen_med'] for r in rows]),
                     np.median([r['imp'] for r in rows]), csv_path))
    else:
        log.info('完成: 0 个被试（全部跳过或无可用数据）')
