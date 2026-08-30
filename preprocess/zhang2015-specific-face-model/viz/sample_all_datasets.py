"""四数据集各抽 10 张归一化可视化（2026-08-30 定稿）

人脸模型均在标准系（解剖轴零位，CLAUDE.md 约定 9）——头姿 pitch/yaw/roll
为解剖角；与 v1 insightface 管线（gen6 原生系）相差 13.46°，头姿读数差
~13° 属坐标系修正而非 bug。pitch 零位 = 眼→鼻连线空间竖直（非自然平视）：
自然平视鼻线前倾 ~13.2°，故自然/微低头姿态读 +1~+16°（正=抬头）；鼻线
前倾是深度倾斜，patch 中只显缩短不显倾斜，勿以 2D 目测校验。
标准系产物已全量核验（canonicalize(true6) 与盘上 true6_canonical 逐点
一致，xgaze 80 + eve 44 人）：
  xgaze/eve    逐人 true6_canonical.txt（严格三角化个性化模型），头姿由
               DLT 3D 点 Kabsch 得到，不走 PnP；xgaze/eve 每帧附跨相机
               HCS gaze 一致性 camΔ（同帧全部相机独立解算，两两最大夹角）；
               EVE 的 face_g_tobii 为 face_R 归一化后的头架向量，需反旋转
               d_cam = -face_R^T·gv(标签) 回相机系（对拍 ±1.000000）；
  gazecapture/ gen_xe6（get_face_model/gen_xe6_canonical.txt：xgaze+EVE
  mpiifacegaze true6 均值，2026-08-30 取代 gen6）+ 6 点 PnP。
归一化统一 normalizeData_face(fixed_forward=False)。

head/gaze 分 raw(原相机系)/norm(归一化虚拟相机系)两列展示（n 在前）；
HCS = hR^T·gc 与归一化无关（R_norm 精确抵消），保持单值。
输出: 本目录 all_datasets_normalized.png（每数据集两行 20 张，行首标注数据集名）
用法（仓库根目录）: python preprocess/zhang2015-specific-face-model/viz/sample_all_datasets.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import scipy.io as sio

_PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_PROJECT / 'preprocess/zhang2015-specific-face-model/get_face_model'))

import face_model_core as core
from utils.logger import get_logger
from utils.normalization import (estimateHeadPose, normalizeData_face,
                                 vector_to_angles)

# 官方 GC 预处理器（dot→CCS→相机系唯一实现，避免手抄漂移）
_gc_spec = importlib.util.spec_from_file_location(
    'gc_pre', _PROJECT / 'preprocess/zhang2015-insightface/gazecapture/preprocessor.py')
_gc_pre = importlib.util.module_from_spec(_gc_spec)
_gc_spec.loader.exec_module(_gc_pre)

log = get_logger('preprocess.specific_face_model.sample_all')

IDX6 = [35, 39, 89, 93, 78, 84]
N_SAMPLE = 20
PER_ROW = 10          # 每数据集两行
OUT = Path(__file__).resolve().parent
W_IMG, H_IMG = 224, 224

# 标准系 gen_xe6：xgaze(80)+EVE(44) true6_canonical 逐点均值再标准化（DLT 真值
# 口径；2026-08-30 取代 gen6 作为 GC/MPII 通用模型——gen6 几何缺陷致 HCS yaw 偏移）
GEN_XE6 = np.loadtxt(
    _PROJECT / 'preprocess/zhang2015-specific-face-model/get_face_model/gen_xe6_canonical.txt')


def _gaze_vec(pitch, yaw):
    return np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch),
                     np.cos(pitch) * np.cos(yaw)])


# --------------------------------------------------------------- XGaze
def load_xgaze(rng):
    LM = Path('/media/yanglinxuan/ylx/xgaze_insightface_224')
    RAW = Path('/media/yanglinxuan/Expansion/xgaze_raw/data/train')
    ANN = Path('/media/yanglinxuan/Expansion/xgaze_raw/data/annotation_train')
    CAL = Path('/media/yanglinxuan/Expansion/xgaze_raw/calibration/cam_calibration')
    FM = Path('/media/yanglinxuan/sfm/xgaze_specific_face_model/face_models')
    FLIP = [3, 6, 13]

    KS, DIST, ROT, TR = {}, {}, {}, {}
    for c in range(18):
        fs = cv2.FileStorage(str(CAL / f'cam{c:02d}.xml'), cv2.FILE_STORAGE_READ)
        KS[c] = fs.getNode('Camera_Matrix').mat()
        DIST[c] = fs.getNode('Distortion_Coefficients').mat()
        ROT[c] = fs.getNode('cam_rotation').mat()
        TR[c] = fs.getNode('cam_translation').mat().reshape(3, 1)
        fs.release()

    out = []
    for sid in rng.permutation(sorted(p.stem for p in LM.glob('subject*.h5'))):
        mpath = FM / sid / 'true6_canonical.txt'   # 标准系个性化模型（已核验）
        if not mpath.is_file():
            continue
        with h5py.File(LM / f'{sid}.h5', 'r') as f:
            fr = f['frame_index'][:].ravel()
            ci = f['cam_index'][:].ravel()
            lm = f['facial_landmarks_2d'][:]
        by_frame = {}
        for r in range(len(fr)):
            by_frame.setdefault(int(fr[r]), []).append((int(ci[r]), r))
        frames = sorted(by_frame)
        rng.shuffle(frames)
        ann = {}                                   # 全部 (frame, cam) 官方注视点
        with open(ANN / f'{sid}.csv') as f:
            for line in f:
                p = line.strip().split(',')
                ann[(p[0], p[1])] = np.array(
                    [float(p[4]), float(p[5]), float(p[6])]).reshape(3, 1)
        dummy = np.zeros((32, 32, 3), np.uint8)
        for fidx in frames:
            rows = by_frame[fidx]
            # DLT：前 10 台相机射线 → 世界系 3D 点；Kabsch 消头运动得头姿
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
            model = np.loadtxt(mpath)
            R_head, t_head = core.kabsch(model, X_w)
            # 跨相机 HCS 一致性：各相机独立（官方逐相机注视点 + 同一头姿）
            gps, hs = {}, []
            for c, r in rows:
                gp_c = ann.get((f'frame{fidx:04d}', f'cam{c:02d}.JPG'))
                if gp_c is None:
                    continue
                rv = cv2.Rodrigues(ROT[c] @ R_head)[0]
                tv = ROT[c] @ t_head.reshape(3, 1) + TR[c]
                _, hr_c, gc_c = normalizeData_face(
                    dummy, model, rv, tv, gp_c, KS[c],
                    fixed_forward=False)[:3]
                gps[c] = gp_c
                hs.append((cv2.Rodrigues(hr_c)[0].T @ gc_c).ravel())
            if len(hs) < 6:
                continue
            hv = np.array(hs); hv /= np.linalg.norm(hv, axis=1, keepdims=True)
            aa = np.degrees(np.arccos(np.clip(hv @ hv.T, -1, 1)))
            camd = aa[np.triu_indices(len(hv), 1)].max()
            # 主相机随机选；world→cam 用官方外参合成 model→cam 变换
            cam, lm_row = rows[rng.integers(len(rows))]
            if cam not in gps:
                continue
            img = cv2.imread(str(RAW / sid / f'frame{fidx:04d}' / f'cam{cam:02d}.JPG'))
            if img is None:
                continue
            if cam in FLIP:
                img = cv2.rotate(img, cv2.ROTATE_180)
            rvec = cv2.Rodrigues(ROT[cam] @ R_head)[0]
            tvec = ROT[cam] @ t_head.reshape(3, 1) + TR[cam]
            out.append((img, lm[lm_row], KS[cam], DIST[cam], model, gps[cam],
                        f'{sid[-4:]}/f{fidx:04d}/c{cam:02d}', rvec, tvec, camd))
            break
        if len(out) >= N_SAMPLE:
            break
    return out


# --------------------------------------------------------------- EVE
def load_eve(rng):
    LM_ROOT = Path('/media/yanglinxuan/zyx/EVE_dataset/eve_dataset/landmarks')
    RAW = Path('/media/yanglinxuan/zyx/EVE_dataset/eve_dataset')
    FM = Path('/media/yanglinxuan/sfm/eve_specific_face_model/face_models')

    subs = [(sp, p.stem) for sp in ('train', 'test')
            for p in sorted(LM_ROOT.joinpath(sp).glob('*.h5'))]
    out = []
    for sp, subj in rng.permutation(subs):
        mpath = FM / subj / 'true6_canonical.txt' # 标准系个性化模型（已核验）
        if not mpath.is_file():
            continue
        with h5py.File(LM_ROOT / sp / f'{subj}.h5', 'r') as f:
            fr = f['frame_index'][:].ravel()
            ci = f['cam_index'][:].ravel()
            st = f['step_index'][:].ravel()
            lm = f['facial_landmarks_2d'][:]
            cameras = json.loads(f.attrs['cameras'])
            steps = json.loads(f.attrs['steps'])
        # 帧同步：basler(30fps)=cam0 帧号/2 对齐 webcam(15fps)
        by_sync = {}
        for r in range(len(fr)):
            c, raw_f, si = int(ci[r]), int(fr[r]), int(st[r])
            sync_f = raw_f // 2 if c == 0 else raw_f
            by_sync.setdefault((sync_f, si), []).append((c, r, raw_f, si))
        groups = [v for v in by_sync.values()
                  if len(set(x[0] for x in v)) >= 3]
        rng.shuffle(groups)
        for rows_raw in groups:
            step_name = steps[rows_raw[0][3]]
            # 该 step 全部相机内参/外参（官方 camera_transformation）
            Ks, Rs, ts, ok_load = {}, {}, {}, True
            for c, cam_name in enumerate(cameras):
                p_h5 = RAW / subj / step_name / f'{cam_name}.h5'
                if not p_h5.is_file():
                    ok_load = False
                    break
                with h5py.File(p_h5, 'r') as f:
                    Ks[c] = np.array(f['camera_matrix'], dtype=float)
                    T = np.array(f['camera_transformation'], dtype=float)
                Rs[c] = T[:3, :3]
                ts[c] = T[:3, 3].reshape(3, 1)
            if not ok_load:
                continue
            rays, pv = [], []
            for c, r, _, _ in rows_raw:
                lm_n = cv2.undistortPoints(
                    lm[r].astype(np.float64).reshape(-1, 1, 2),
                    Ks[c], None).reshape(-1, 2)
                rays.append(lm_n[core.IDX6])
                pv.append(np.concatenate([cv2.Rodrigues(Rs[c])[0].ravel(),
                                          ts[c].ravel()]))
            X_w = core.triangulate(np.stack(rays), np.stack(pv), n_points=6)
            model = np.loadtxt(mpath)
            R_head, t_head = core.kabsch(model, X_w)
            # 官方 PoG 直算：屏幕 px → 相机系 3D 注视点（dataset_report §4；
            # 各相机标注为同一 tobii 流按同步帧号分发，跨相机一致 ~0.04°）
            dummy = np.zeros((32, 32, 3), np.uint8)
            gps, hs = {}, []
            for c, r, raw_f, _ in rows_raw:
                p_h5 = RAW / subj / step_name / f'{cameras[c]}.h5'
                with h5py.File(p_h5, 'r') as f:
                    if raw_f >= len(f['face_PoG_tobii/validity']) or \
                            not f['face_PoG_tobii/validity'][raw_f]:
                        continue
                    PoG = np.array(f['face_PoG_tobii/data'][raw_f])
                    mmpp = np.array(f['millimeters_per_pixel'], dtype=float)
                    T = np.array(f['camera_transformation'], dtype=float)
                gp_c = (T @ np.array([PoG[0] * mmpp[0], PoG[1] * mmpp[1],
                                      0., 1.]))[:3].reshape(3, 1)
                rv = cv2.Rodrigues(Rs[c] @ R_head)[0]
                tv = Rs[c] @ t_head.reshape(3, 1) + ts[c]
                _, hr_c, gc_c = normalizeData_face(
                    dummy, model, rv, tv, gp_c, Ks[c],
                    fixed_forward=False)[:3]
                gps[c] = gp_c
                hs.append((cv2.Rodrigues(hr_c)[0].T @ gc_c).ravel())
            if set(gps) != set(range(4)):
                continue     # 组门控：四相机齐全且 PoG 全有效，缺一即弃
            hv = np.array(hs); hv /= np.linalg.norm(hv, axis=1, keepdims=True)
            aa = np.degrees(np.arccos(np.clip(hv @ hv.T, -1, 1)))
            camd = aa[np.triu_indices(len(hv), 1)].max()
            cam_c, lm_row, fidx, _ = rows_raw[rng.integers(len(rows_raw))]
            if cam_c not in gps:
                continue
            cam_name = cameras[cam_c]
            cap = cv2.VideoCapture(str(RAW / subj / step_name / f'{cam_name}.mp4'))
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, img = cap.read()
            cap.release()
            if not ok or img is None:
                continue
            rvec = cv2.Rodrigues(Rs[cam_c] @ R_head)[0]
            tvec = Rs[cam_c] @ t_head.reshape(3, 1) + ts[cam_c]
            out.append((img, lm[lm_row], Ks[cam_c], None, model, gps[cam_c],
                        f'{subj}/{cam_name}/f{fidx:04d}', rvec, tvec, camd))
            break
        if len(out) >= N_SAMPLE:
            break
    return out


# --------------------------------------------------------------- GazeCapture
def load_gc(rng):
    LM = Path('/media/yanglinxuan/zyx/GazeCapture/landmarks')
    RAW = Path('/media/yanglinxuan/zyx/GazeCapture')
    CAL = Path('/media/yanglinxuan/zyx/GazeCapture/calibration')

    subs = [(sp, p.stem) for sp in ('train', 'test')
            for p in sorted(LM.joinpath(sp).glob('*.h5'))]
    out = []
    for sp, sess in rng.permutation(subs):
        with h5py.File(LM / sp / f'{sess}.h5', 'r') as f:
            fr = f['frame_index'][:].ravel()
            ori = f['orientation'][:].ravel()
            lm = f['facial_landmarks_2d'][:]
        r = int(rng.integers(len(fr)))
        fidx, ori, lm106 = int(fr[r]), int(ori[r]), lm[r]
        img = cv2.imread(str(RAW / sess / 'frames' / f'{fidx:05d}.jpg'))
        if img is None:
            continue
        try:
            device = json.load(open(RAW / sess / 'info.json'))['DeviceName']
            dot = json.load(open(RAW / sess / 'dotInfo.json'))
            pos = {int(n.split('.')[0]): i for i, n in
                   enumerate(json.load(open(RAW / sess / 'frames.json')))}
        except Exception:
            continue
        pi = pos.get(fidx)
        if pi is None or dot['DotNum'][pi] == -1:
            continue
        w, h = (480, 640) if ori in (1, 2) else (640, 480)
        cal = CAL / f"{device.lower().replace(' ', '-')}_{w}x{h}.xml"
        if not cal.is_file():
            continue
        fs = cv2.FileStorage(str(cal), cv2.FILE_STORAGE_READ)
        K = fs.getNode('Camera_Matrix').mat()
        dist = fs.getNode('Distortion_Coefficients').mat()
        fs.release()
        # 官方链路直调：dot→CCS（过 invalid_dot 门）→相机系
        xc, yc = dot['XCam'][pi], dot['YCam'][pi]
        ccs_x, ccs_y = _gc_pre._dot_to_ccs_mm(ori, xc, yc)
        if ccs_y <= 0:
            continue                         # invalid_dot（朝向过渡帧噪声）
        gp = np.array(_gc_pre._gaze_point_cam(ori, ccs_x, ccs_y))
        out.append((img, lm106, K, dist, GEN_XE6, gp.reshape(3, 1),
                    f'{sess}/f{fidx:05d}/o{ori}', None, None, None))
        if len(out) >= N_SAMPLE:
            break
    return out


# --------------------------------------------------------------- MPIIFaceGaze
def load_mpii(rng):
    LM = Path('/media/yanglinxuan/zyx/MPIIFaceGaze/landmarks')
    RAW = Path('/media/yanglinxuan/zyx/MPIIFaceGaze')

    subs = sorted(p.stem for p in LM.glob('*.h5'))     # 仅 15 人
    data = {}
    for subj in subs:
        with h5py.File(LM / f'{subj}.h5', 'r') as f:
            lm = f['facial_landmarks_2d'][:]
            names = [n.decode() if isinstance(n, bytes) else str(n)
                     for n in f['image_name'][:]]
            days = f['day_index'][:].ravel()
        mat = sio.loadmat(RAW / subj / 'Calibration' / 'Camera.mat')
        data[subj] = (lm, names, days,
                      np.array(mat['cameraMatrix'], dtype=float),
                      np.array(mat['distCoeffs'], dtype=float).ravel())

    out, used = [], set()
    def one(subj):
        lm, names, days, K, dist = data[subj]
        for _ in range(10):                            # 每次最多试 10 帧
            r = int(rng.integers(len(lm)))
            day, fname = int(days[r]), names[r]
            if (subj, fname) in used:
                continue
            img = cv2.imread(str(RAW / subj / f'day{day:02d}' / fname))
            if img is None:
                continue
            gp = None
            with open(RAW / subj / f'{subj}.txt') as f:   # 列 24-26 = gaze
                for line in f:
                    p = line.split()
                    if p[0] == f'day{day:02d}/{fname}':
                        gp = np.array([float(p[24]), float(p[25]),
                                       float(p[26])]).reshape(3, 1)
                        break
            if gp is None:
                continue
            used.add((subj, fname))
            out.append((img, lm[r], K, dist, GEN_XE6, gp,
                        f'{subj}/d{day:02d}', None, None, None))
            return True
        return False

    for subj in rng.permutation(subs):                 # 第一轮：每人一张
        if one(subj) and len(out) >= N_SAMPLE:
            break
    fail = 0
    while len(out) < N_SAMPLE and fail < 100:          # 第二轮：补齐到 20
        if not one(subs[int(rng.integers(len(subs)))]):
            fail += 1
    return out


# --------------------------------------------------------------- 主流程
def process_one(item):
    img, lm106, K, dist, model6, gp, sample_id, rvec, tvec, camd = item
    if dist is None:
        dist = np.zeros((1, 5))
    if rvec is None:                       # GC/MPII：gen6 6 点 PnP
        rvec, tvec = estimateHeadPose(
            lm106[IDX6].reshape(6, 1, 2).astype(float), model6, K, dist)
    img_w, hr_norm, gc_ccs = normalizeData_face(
        img, model6, rvec, tvec, gp, K, fixed_forward=False)[:3]
    hR = cv2.Rodrigues(hr_norm)[0]
    # raw（原相机系）：头姿直接由 rvec，视线 = gp−face_center（未经 R_norm）
    hR_raw = cv2.Rodrigues(rvec)[0]
    fc = (hR_raw @ model6.T + tvec.reshape(3, 1)).mean(1)
    gc_raw = gp.reshape(3) - fc
    gc_raw /= np.linalg.norm(gc_raw)
    # HCS = hR_norm^T·gc_norm = hR_raw^T·gc_raw（R_norm 精确抵消，与归一化无关）
    gc_hcs = hR.T @ gc_ccs
    fwd = np.array([0.0, 0.0, -1.0])
    return {'patch': img_w, 'id': sample_id,
            'head': np.degrees(vector_to_angles(hR @ fwd)),
            'head_r': np.degrees(vector_to_angles(hR_raw @ fwd)),
            'ccs': np.degrees(vector_to_angles(gc_ccs.ravel())),
            'ccs_r': np.degrees(vector_to_angles(gc_raw)),
            'hcs': np.degrees(vector_to_angles(gc_hcs.ravel())),
            'camd': camd}


def main():
    rng = np.random.default_rng(42)
    rows = [('xgaze', 'true6', load_xgaze), ('eve', 'true6', load_eve),
            ('gazecapture', 'gen_xe6', load_gc), ('mpiifacegaze', 'gen_xe6', load_mpii)]

    all_rows = []
    for name, tag, loader in rows:
        log.info(f'加载 {name}...')
        res = []
        for item in loader(rng):
            try:
                res.append(process_one(item))
            except Exception as e:
                log.warning(f'{name} 某样本失败: {e}')
        all_rows.append((name, tag, res[:N_SAMPLE]))
        log.info(f'{name}: {len(res[:N_SAMPLE])} 样本')

    # 出图：每数据集两行 20 张；块间留白分隔；左侧粗体标签（自适应宽度）
    LABEL_W, PAD, BLOCK_GAP = 185, 6, 30
    CELL_H = H_IMG + 98
    N_ROWS = (N_SAMPLE + PER_ROW - 1) // PER_ROW
    W = LABEL_W + PER_ROW * (W_IMG + PAD) + PAD
    H = len(all_rows) * (N_ROWS * CELL_H + BLOCK_GAP) - BLOCK_GAP + PAD + 26
    canvas = np.full((H, W, 3), 255, np.uint8)
    cv2.putText(canvas, 'n = normalized (virtual cam, first col) | '
                'r = raw (original camera frame) | HCS: normalization-invariant',
                (12, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (60, 60, 60), 1, cv2.LINE_AA)

    TOP = 26
    for di, (name, tag, res) in enumerate(all_rows):
        y_top = TOP + PAD + di * (N_ROWS * CELL_H + BLOCK_GAP)
        img_h = (N_ROWS - 1) * CELL_H + W_IMG           # 两行「图片区」总高
        yc = y_top + img_h // 2                         # 标签对齐图片区中心
        if di > 0:                                      # 数据集分隔线
            cv2.line(canvas, (0, y_top - BLOCK_GAP // 2),
                     (W, y_top - BLOCK_GAP // 2), (200, 200, 200), 2)
        cv2.rectangle(canvas, (0, y_top),               # 标签底色带（同图片区）
                      (LABEL_W - 10, y_top + img_h - 1),
                      (232, 238, 246), -1)
        fs = 0.95                                       # 自适应字号
        while fs > 0.3 and cv2.getTextSize(
                name, cv2.FONT_HERSHEY_SIMPLEX, fs, 3)[0][0] > LABEL_W - 24:
            fs -= 0.05
        cv2.putText(canvas, name, (12, yc - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (25, 25, 110), 3, cv2.LINE_AA)
        cv2.putText(canvas, f'({tag})', (12, yc + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (90, 90, 90), 2, cv2.LINE_AA)
        for i, r in enumerate(res):
            rr, cc = divmod(i, PER_ROW)
            x0 = LABEL_W + cc * (W_IMG + PAD)
            y0 = y_top + rr * CELL_H          # 与标签同用块坐标（含 BLOCK_GAP）
            canvas[y0:y0 + H_IMG, x0:x0 + W_IMG] = r['patch']
            put = lambda txt, dy, c, dx=2: cv2.putText(
                canvas, txt, (x0 + dx, y0 + H_IMG + dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA)
            put(f"{r['id'][:17]}", 16, (90, 90, 90))
            put('head', 36, (120, 120, 120))
            put(f"n({r['head'][0]:+5.1f},{r['head'][1]:+5.1f})", 36, (30, 30, 30), 44)
            put(f"r({r['head_r'][0]:+5.1f},{r['head_r'][1]:+5.1f})", 36, (30, 30, 30), 130)
            put('gaze', 56, (120, 120, 120))
            put(f"n({r['ccs'][0]:+5.1f},{r['ccs'][1]:+5.1f})", 56, (0, 0, 200), 44)
            put(f"r({r['ccs_r'][0]:+5.1f},{r['ccs_r'][1]:+5.1f})", 56, (0, 0, 200), 130)
            put(f"HCS  ({r['hcs'][0]:+5.1f},{r['hcs'][1]:+5.1f})", 76, (0, 140, 0))
            if r['camd'] is not None:
                put(f"camd {r['camd']:5.2f} deg", 96, (0, 120, 255))

    cv2.imwrite(str(OUT / 'all_datasets_normalized.png'), canvas)
    log.info(f'输出 {OUT / "all_datasets_normalized.png"} '
             f'({len(all_rows)} 数据集 × {N_SAMPLE} 张)')


if __name__ == '__main__':
    main()
