"""导出 xgaze Raw Head Pose pitch≈60° 拖尾样本供人工研究（2026-09-02）

背景：gaze_distribution_specific.png 中 xgaze 的 Raw Head Pose 在 pitch≈60°
出现从 yaw=0 指向 yaw=+90° 的拖尾。实测（8 被试 × 200 帧全量 DLT）：
pitch>45° 的样本 **全部来自 cam13**（elev +83.4°/azim −127.0°，近正上方），
占 2.8%；其余 17 台相机零贡献。本脚本把拖尾样本连图带数导出到
viz/distribution/exception/pitch60_tail/，每个 case 含：
  - raw_cam13.jpg   cam13 原始帧（缩放 1600px，106 特征点 + 6 建模点 + 读数标注）
  - raw_cam00.jpg   同帧正前方相机（头姿上下文对照）
  - norm_cam13.png  v2 归一化 patch（sfm h5）+ 归一化头姿/视线标注
  - 汇总 data.json + README.md（含该帧全部 18 相机读数表）

用法：直接运行（扫描前 N_SCAN 个被试，全自动选样导出）。
"""
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parents[3]
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_PROJECT / 'preprocess' / 'zhang2015-specific-face-model'))
sys.path.insert(0, str(_PROJECT / 'preprocess' / 'zhang2015-specific-face-model' / 'get_face_model'))

import cv2
import h5py
import numpy as np

import face_model_core as core
from utils.normalization import vector_to_angles, HEAD_PITCH_OFFSET

# ---------------------------------------------------------------- 常量
CAL = Path('/media/yanglinxuan/Expansion/xgaze_raw/calibration/cam_calibration')
RAW = Path('/media/yanglinxuan/Expansion/xgaze_raw/data/train')
ANN = Path('/media/yanglinxuan/Expansion/xgaze_raw/data/annotation_train')
LM1 = Path('/media/yanglinxuan/ylx/xgaze_insightface_224')       # v1 特征点（分布图同源）
V2 = Path('/media/yanglinxuan/sfm/xgaze_specific_224')           # v2 归一化产物
FM = Path('/media/yanglinxuan/sfm/xgaze_specific_face_model/face_models')
OUT = _HERE / 'exception' / 'pitch60_tail'

N_SCAN = 16                 # 扫描被试数（每人 200 帧全量）
PITCH_LO, PITCH_HI = 50.0, 70.0    # cam13 raw pitch 拖尾选样窗口
YAW_BINS = list(range(0, 100, 10))  # yaw 分箱选样，覆盖 0→+90
N_CONTROL = 2               # 对照组：cam13 pitch 20–35°
OUT_W = 1600                # 原始图导出宽度（全分辨率图过大，仅缩放存档）
# v1 管线对这几台相机先 180° 旋转再做关键点检测（见 normalize_xgaze.py），
# h5 中的 facial_landmarks_2d 是旋转后坐标——画到未旋转的原始图上需反映射
FLIP_CAMERAS = [3, 6, 13]
# cam13 外参（官方标定，标注到图上；欧拉角为人类可读形式）
_fs13 = cv2.FileStorage(str(CAL / 'cam13.xml'), cv2.FILE_STORAGE_READ)
ROT13 = _fs13.getNode('cam_rotation').mat()
TR13 = _fs13.getNode('cam_translation').mat().reshape(3, 1)
_fs13.release()
# 欧拉角（scipy extrinsic 'xyz'：R = Rz(gamma)·Ry(beta)·Rx(alpha)，固定轴 x→y→z）
from scipy.spatial.transform import Rotation
ALPHA13, BETA13, GAMMA13 = Rotation.from_matrix(ROT13).as_euler(
    'xyz', degrees=True)
POS13 = (-ROT13.T @ TR13).ravel()      # 相机在世界系的位置（mm）


def load_calib():
    ks, dist, rot, tr = {}, {}, {}, {}
    for c in range(18):
        fs = cv2.FileStorage(str(CAL / f'cam{c:02d}.xml'), cv2.FILE_STORAGE_READ)
        ks[c] = fs.getNode('Camera_Matrix').mat()
        dist[c] = fs.getNode('Distortion_Coefficients').mat()
        rot[c] = fs.getNode('cam_rotation').mat()
        tr[c] = fs.getNode('cam_translation').mat().reshape(3, 1)
        fs.release()
    return ks, dist, rot, tr


def raw_head(rot, tr, c, r_head):
    """c 相机看 r_head 头姿的 raw (pitch, yaw)，与分布图 HEAD_RAW 同口径"""
    rvec = cv2.Rodrigues(rot[c] @ r_head)[0]
    v = cv2.Rodrigues(rvec)[0] @ np.array([0., 0., -1.])
    t, p = vector_to_angles(v)
    return float(np.degrees(t) + HEAD_PITCH_OFFSET), float(np.degrees(p))


def scan(ks, dist, rot, tr):
    """返回候选列表：[{subject, frame, cam13_pitch/yaw, all_cams, elev, azim}]"""
    subs = sorted(p.stem for p in LM1.glob('subject*.h5'))[:N_SCAN]
    cands, elevs_all = [], []
    for sid in subs:
        model = np.loadtxt(FM / sid / 'true6_canonical.txt')
        with h5py.File(LM1 / f'{sid}.h5', 'r') as f:
            fr = f['frame_index'][:].ravel()
            ci = f['cam_index'][:].ravel()
            lm = f['facial_landmarks_2d'][:]
        by_frame = {}
        for r in range(len(fr)):
            by_frame.setdefault(int(fr[r]), []).append((int(ci[r]), r))
        for fidx in sorted(by_frame):
            rows = by_frame[fidx]
            rays, pv = [], []
            for c, r in rows:
                if c >= 10:
                    continue
                lm_n = cv2.undistortPoints(
                    lm[r].astype(np.float64).reshape(-1, 1, 2),
                    ks[c], dist[c]).reshape(-1, 2)
                rays.append(lm_n[core.IDX6])
                pv.append(np.concatenate([cv2.Rodrigues(rot[c])[0].ravel(),
                                          tr[c].ravel()]))
            if len(rays) < 6:
                continue
            X_w = core.triangulate(np.stack(rays), np.stack(pv), n_points=6)
            R_head, _ = core.kabsch(model, X_w)
            v_w = R_head @ np.array([0., 0., -1.])
            elev = float(np.degrees(np.arcsin(v_w[1] / np.linalg.norm(v_w))))
            az = float(np.degrees(np.arctan2(v_w[0], -v_w[2])) - 180.0)
            az = (az + 180) % 360 - 180
            elevs_all.append(elev)
            all_c = {f'cam{c:02d}': raw_head(rot, tr, c, R_head) for c, _ in rows}
            key = 'cam13'
            if key not in all_c:
                continue
            p13, y13 = all_c[key]
            cands.append({'subject': sid, 'frame': fidx, 'cam13_pitch': p13,
                          'cam13_yaw': y13, 'all_cams': all_cams_dump(all_c),
                          'elev': elev, 'azim': az})
    nat = float(np.median(elevs_all))
    for c_ in cands:
        c_['elev_vs_natural'] = c_['elev'] - nat
    return cands, nat


def all_cams_dump(all_c):
    return {k: [round(v[0], 2), round(v[1], 2)] for k, v in all_c.items()}


def select(cands):
    """yaw 0→+90 分箱选拖尾样本 + 对照组"""
    picked = []
    used = set()
    for y0 in YAW_BINS:
        pool = [c_ for c_ in cands
                if PITCH_LO <= c_['cam13_pitch'] <= PITCH_HI
                and y0 <= c_['cam13_yaw'] < y0 + 10
                and (c_['subject'], c_['frame']) not in used]
        if not pool:
            continue
        # 箱内优先 pitch 最接近 60
        best = min(pool, key=lambda c_: abs(c_['cam13_pitch'] - 60.0))
        picked.append(best)
        used.add((best['subject'], best['frame']))
    ctrls = []
    pool = [c_ for c_ in cands
            if 20 <= c_['cam13_pitch'] <= 35
            and (c_['subject'], c_['frame']) not in used]
    if pool:
        pool.sort(key=lambda c_: c_['cam13_yaw'])
        for i in np.linspace(0, len(pool) - 1, N_CONTROL).astype(int):
            ctrls.append(pool[int(i)])
    return picked, ctrls


# ---------------------------------------------------------------- 绘图
def annotate_raw(img, lm106, idx6, lines):
    """缩放 + 画点 + 右下角文本，返回导出图"""
    h, w = img.shape[:2]
    s = OUT_W / w
    out = cv2.resize(img, (OUT_W, int(h * s)))
    r = max(2, int(OUT_W / 500))
    for i, (x, y) in enumerate(lm106):
        p = (int(x * s), int(y * s))
        cv2.circle(out, p, r, (0, 220, 0), -1)
    for i in idx6:
        x, y = lm106[i]
        cv2.circle(out, (int(x * s), int(y * s)), r * 2, (0, 0, 255), 2)
    fs = max(0.5, OUT_W / 1400)
    th = max(1, int(OUT_W / 800))
    lh = int(30 * OUT_W / 1000)
    mg = int(12 * OUT_W / 1000)
    H = out.shape[0]
    for k, ln in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        org = (OUT_W - tw - mg, H - mg - (len(lines) - 1 - k) * lh)
        cv2.putText(out, ln, org, cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.putText(out, ln, org, cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (80, 255, 255), th, cv2.LINE_AA)
    return out


def annotate_norm(patch, lines):
    """224 patch 放大 + 下方文本条（右对齐，字号自适应防截断）"""
    big = cv2.resize(patch, (448, 448), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((448 + 170, 448, 3), 24, np.uint8)
    canvas[:448] = big
    fs = 0.55
    while fs > 0.3 and max(
            cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0]
            for ln in lines) > 428:
        fs -= 0.02
    for k, ln in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        org = (448 - tw - 10, 448 + 30 + k * 28)
        cv2.putText(canvas, ln, org, cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (230, 230, 230), 1, cv2.LINE_AA)
    return canvas


# ---------------------------------------------------------------- 导出
def load_v2_index(sid):
    """v2 h5 的 (frame, cam) → 行号 与 数据组"""
    with h5py.File(V2 / f'{sid}.h5', 'r') as f:
        fr = f['frame_index'][:].ravel()
        ci = f['cam_index'][:].ravel()
        idx = {(int(fr[i]), int(ci[i])): i for i in range(len(fr))}
        return idx, f


def official_dot(sid, fidx, cam_name):
    """官方 csv 的注视点屏幕坐标 (dot_x, dot_y)"""
    key_frame, key_file = f'frame{fidx:04d}', f'{cam_name}.JPG'
    with open(ANN / f'{sid}.csv') as f:
        for line in f:
            p = line.strip().split(',')
            if p[0] == key_frame and p[1] == key_file:
                return [float(p[2]), float(p[3])]
    return None


def export_case(nn, cand, is_ctrl):
    sid, fidx = cand['subject'], cand['frame']
    fdir = RAW / sid / f'frame{fidx:04d}'
    tag = 'ctrl' if is_ctrl else 'tail'
    case_dir = OUT / f'{nn:02d}_{tag}_{sid}_f{fidx:04d}_cam13'
    case_dir.mkdir(parents=True, exist_ok=True)

    # v1 特征点（该帧各相机）
    with h5py.File(LM1 / f'{sid}.h5', 'r') as f:
        fr = f['frame_index'][:].ravel()
        ci = f['cam_index'][:].ravel()
        lm = f['facial_landmarks_2d'][:]
    lm_of = {}
    for r in range(len(fr)):
        if int(fr[r]) == fidx:
            lm_of[int(ci[r])] = lm[r]

    # 原始图：cam13（拖尾读数） + cam00（正前方对照）
    for cam, fname in ((13, 'raw_cam13.jpg'), (0, 'raw_cam00.jpg')):
        img = cv2.imread(str(fdir / f'cam{cam:02d}.JPG'))
        if img is None:
            continue
        lm_draw = lm_of[cam]
        if cam in FLIP_CAMERAS:      # 特征点在旋转 180° 后的坐标系，反映射回原始图
            h, w = img.shape[:2]
            lm_draw = np.stack([w - 1 - lm_draw[:, 0],
                                h - 1 - lm_draw[:, 1]], axis=1)
        p, y = cand['all_cams'][f'cam{cam:02d}']
        if cam == 13:
            lines = [
                f'{sid} frame{fidx:04d} cam13',
                f'RAW head: pitch {cand["cam13_pitch"]:+.1f} yaw {cand["cam13_yaw"]:+.1f}  <== tail',
                f'extr: alpha {ALPHA13:+.1f} beta {BETA13:+.1f} gamma {GAMMA13:+.1f} deg (xyz)',
                'pos: [{:+.0f} {:+.0f} {:+.0f}] mm  elev +83.4 azim -127.0'.format(*POS13),
            ]
        else:
            lines = [
                f'{sid} frame{fidx:04d} cam00 (front, same frame)',
                f'RAW head: pitch {p:+.1f} yaw {y:+.1f}  (normal)',
            ]
        cv2.imwrite(str(case_dir / fname),
                    annotate_raw(img, lm_draw, core.IDX6, lines))

    # v2 归一化 patch + 读数
    v2_idx, _ = load_v2_index(sid)
    rec = dict(cand)
    i = v2_idx.get((fidx, 13))
    if i is not None:
        with h5py.File(V2 / f'{sid}.h5', 'r') as f:
            patch = f['face_patch'][i]
            hp = f['face_head_pose'][i]
            gz = f['face_gaze'][i]
            gh = f['face_gaze_hcs'][i]
        lines = [
            f'{sid} frame{fidx:04d} cam13 normalized (v2)',
            f'RAW head: pitch {cand["cam13_pitch"]:+.1f} yaw {cand["cam13_yaw"]:+.1f}  <== tail',
        ]
        cv2.imwrite(str(case_dir / 'norm_cam13.png'), annotate_norm(patch, lines))
        rec['v2'] = {'head': [round(float(x), 2) for x in hp],
                     'gaze_ccs': [round(float(x), 2) for x in gz],
                     'gaze_hcs': [round(float(x), 2) for x in gh]}
    return case_dir.name, rec


def main():
    global NAT_MEDIAN
    ks, dist, rot, tr = load_calib()
    OUT.mkdir(parents=True, exist_ok=True)

    if '--redraw' in sys.argv and (OUT / 'data.json').exists():
        # 仅重画：case 数据取自已归档的 data.json，不重扫
        db = json.load(open(OUT / 'data.json'))
        NAT_MEDIAN = db['natural_elev_median']
        for i, rec in enumerate(db['cases'], 1):
            is_ctrl = rec['cam13_pitch'] < PITCH_LO   # 对照组 pitch 远低于拖尾窗口
            name, _ = export_case(i, rec, is_ctrl=is_ctrl)
            print(f'  [{i:02d}] redrawn {name}')
        print(f'redrawn {len(db["cases"])} cases -> {OUT}')
        return

    print(f'scanning {N_SCAN} subjects ...')
    cands, nat = scan(ks, dist, rot, tr)
    NAT_MEDIAN = nat
    tail_pool = [c_ for c_ in cands
                 if PITCH_LO <= c_['cam13_pitch'] <= PITCH_HI]
    print(f'natural elev median {nat:+.1f}; tail candidates {len(tail_pool)} '
          f'(pitch {PITCH_LO}-{PITCH_HI})')
    for c_ in cands:                       # 补充官方 csv 注视点屏幕坐标
        c_['dot'] = official_dot(c_['subject'], c_['frame'], 'cam13')
    picked, ctrls = select(cands)
    records = []
    nn = 1
    for c_ in picked:
        name, rec = export_case(nn, c_, is_ctrl=False)
        records.append(rec)
        print(f'  [{nn:02d}] {name}  pitch {c_["cam13_pitch"]:+.1f} yaw {c_["cam13_yaw"]:+.1f}')
        nn += 1
    for c_ in ctrls:
        name, rec = export_case(nn, c_, is_ctrl=True)
        records.append(rec)
        print(f'  [{nn:02d}] {name}  pitch {c_["cam13_pitch"]:+.1f} yaw {c_["cam13_yaw"]:+.1f} (control)')
        nn += 1
    json.dump({'natural_elev_median': nat, 'pitch_window': [PITCH_LO, PITCH_HI],
               'scan_subjects': N_SCAN, 'cases': records},
              open(OUT / 'data.json', 'w'), indent=1, ensure_ascii=False)
    print(f'done: {len(records)} cases -> {OUT}')


if __name__ == '__main__':
    main()
