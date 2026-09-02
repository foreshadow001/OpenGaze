"""四数据集 v3 归一化可视化（ours-without-roll，2026-09-02）

**采样与 v2 版完全一致**（同被试/同帧/同相机/同原图字节/同模型/同注视点）：
直接 import 并调用 v2 版
[../zhang2015-specific-face-model/viz/sample_all_datasets.py](../zhang2015-specific-face-model/viz/sample_all_datasets.py)
的四个 loader（同一 rng=42、同一顺序、同一门控），v2 的 item 携带原始图、
K、模型、rvec/tvec、注视点——v3 只把归一化换成
`normalizeData_face(fixed_forward=True)`（roll-only，协议见 ../normalization_protocol.md）。
两张图（v2/v3）可逐面板对照：同一来源，仅归一化不同。

head 行展示**欧拉角**（extrinsic xyz：α=pitch, β=yaw, γ=roll）而非方向读数：
fixed_forward 把模型 −z 轴转到像平面水平方向，head_pose_angles 的方向读数在
归一化帧退化为 pitch≡−30°；欧拉角才是协议中"pitch/yaw 归一化前后不变、
roll=0"的严格量——n 列与 r 列的 (α,β) 应逐位相等，即本图的内置校验。
gaze 行 n=v3 归一化相机系（roll 修正、头姿保留），r=原相机系；HCS 单值
不变（与 v2 图逐位同）。wld 行 = 世界系头姿 (elev, azim)（协议
face_head_elev_azim 口径：elev 加 −30° 零位）——世界系：xgaze=官方穹顶系
（=cam00）、EVE=basler 相机系（cam 0，由 v2 loader 附带的 step 取外参）、
GC/MPII=相机本身；多相机同帧各相机反解同值。

排版与 v2 版 sample_all_datasets.py 一致（每数据集两行 20 张 + 行首标签）。
输出: 本目录 all_datasets_normalized.png
用法（仓库根目录）: python preprocess/ours-without-roll/viz/sample_all_datasets.py
"""
import importlib.util
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

_PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_PROJECT / 'preprocess/zhang2015-specific-face-model/get_face_model'))

from utils.logger import get_logger
from utils.normalization import (HEAD_PITCH_OFFSET, estimateHeadPose,
                                 normalizeData_face, vector_to_angles)

# v2 版采样器（唯一采样真源；模块级仅加载常量，无副作用）
_v2_spec = importlib.util.spec_from_file_location(
    'sample_all_v2',
    _PROJECT / 'preprocess/zhang2015-specific-face-model/viz/sample_all_datasets.py')
v2 = importlib.util.module_from_spec(_v2_spec)
_v2_spec.loader.exec_module(v2)

log = get_logger('preprocess.ours_without_roll.sample_all')

IDX6 = v2.IDX6                      # 与 v2 同一关键点索引
N_SAMPLE = 20
PER_ROW = 10          # 每数据集两行
OUT = Path(__file__).resolve().parent
W_IMG, H_IMG = 224, 224

# ---- 世界系外参 ----
_CAL = Path('/media/yanglinxuan/Expansion/xgaze_raw/calibration/cam_calibration')
_ROT = {}
for _c in range(18):
    _fs = cv2.FileStorage(str(_CAL / f'cam{_c:02d}.xml'), cv2.FILE_STORAGE_READ)
    _ROT[_c] = _fs.getNode('cam_rotation').mat()
    _fs.release()

_EVE_RAW = Path('/media/yanglinxuan/zyx/EVE_dataset/eve_dataset')
_eve_ext_cache = {}


def _ext_xgaze(sample_id):
    """id '0063/f0883/c05' → 世界←相机 = ROT_c^T（官方穹顶系）"""
    cam = int(sample_id.rsplit('/c', 1)[1])
    return _ROT[cam].T


def _ext_eve(sample_id, step_name):
    """id 'train33/webcam_l/f0234' → 世界←相机 = R_basler·R_c^T（basler 系）"""
    subj, cam_name, ftag = sample_id.split('/')
    key = (subj, step_name, cam_name)
    if key not in _eve_ext_cache:
        d = _EVE_RAW / subj / step_name
        with h5py.File(d / 'basler.h5', 'r') as f:
            R_b = np.array(f['camera_transformation'], dtype=float)[:3, :3]
        with h5py.File(d / f'{cam_name}.h5', 'r') as f:
            R_c = np.array(f['camera_transformation'], dtype=float)[:3, :3]
        _eve_ext_cache[key] = R_b @ R_c.T
    return _eve_ext_cache[key]


# --------------------------------------------------------------- 采样（v2 真源）
def wrap_xgaze(rng):
    for item in v2.load_xgaze(rng):
        sid = item[6]
        yield item + (_ext_xgaze(sid),)


def wrap_eve(rng):
    for item in v2.load_eve(rng):
        step_name = item[10]                    # v2 loader 末位附带
        yield item[:10] + (_ext_eve(item[6], step_name),)


def wrap_gc(rng):
    for item in v2.load_gc(rng):
        yield item + (None,)                    # 世界=相机


def wrap_mpii(rng):
    for item in v2.load_mpii(rng):
        yield item + (None,)


# --------------------------------------------------------------- v3 处理
def process_one(item):
    img, lm106, K, dist, model6, gp, sample_id, rvec, tvec, camd, ext = item
    if rvec is None:                       # GC/MPII：gen_xe6 6 点 PnP（与 v2 同码）
        rvec, tvec = estimateHeadPose(
            lm106[IDX6].reshape(6, 1, 2).astype(float), model6, K, dist)
    img_w, hr_norm, gc_ccs = normalizeData_face(
        img, model6, rvec, tvec, gp, K, fixed_forward=True)[:3]
    hR = cv2.Rodrigues(rvec)[0]
    hR_n = cv2.Rodrigues(hr_norm)[0]
    # 欧拉（extrinsic xyz）：n 列 (α,β) 应与 r 列逐位相等、γ 归零
    e_n = Rotation.from_matrix(hR_n).as_euler('xyz', degrees=True)
    e_r = Rotation.from_matrix(hR).as_euler('xyz', degrees=True)
    # 原相机系视线 = gp − face_center（与 v2 图 r 列同式）
    fc = (hR @ model6.T + tvec.reshape(3, 1)).mean(1)
    gc_raw = gp.reshape(3) - fc
    gc_raw /= np.linalg.norm(gc_raw)
    gc_hcs = hR_n.T @ gc_ccs                # ≡ hR^T·gp_dir，与归一化无关
    # 世界系头姿（协议 face_head_elev_azim）；ext=None 时世界=相机
    R_hw = ext @ hR if ext is not None else hR
    t_w, p_w = vector_to_angles(R_hw @ np.array([0., 0., -1.]))
    return {'patch': img_w, 'id': sample_id,
            'head': (e_n[0], e_n[1]), 'head_r': (e_r[0], e_r[1]),
            'roll_raw': e_r[2],
            'ccs': np.degrees(vector_to_angles(gc_ccs.ravel())),
            'ccs_r': np.degrees(vector_to_angles(gc_raw)),
            'hcs': np.degrees(vector_to_angles(gc_hcs.ravel())),
            'world': (np.degrees(t_w) + HEAD_PITCH_OFFSET, np.degrees(p_w)),
            'camd': camd}


def main():
    rng = np.random.default_rng(42)         # 与 v2 版同种子、同调用顺序
    rows = [('xgaze', 'true6', wrap_xgaze), ('eve', 'true6', wrap_eve),
            ('gazecapture', 'gen_xe6', wrap_gc), ('mpiifacegaze', 'gen_xe6', wrap_mpii)]

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
    CELL_H = H_IMG + 118
    N_ROWS = (N_SAMPLE + PER_ROW - 1) // PER_ROW
    W = LABEL_W + PER_ROW * (W_IMG + PAD) + PAD
    H = len(all_rows) * (N_ROWS * CELL_H + BLOCK_GAP) - BLOCK_GAP + PAD + 26
    canvas = np.full((H, W, 3), 255, np.uint8)
    cv2.putText(canvas, 'v3 ours-without-roll (fixed_forward=True) | '
                'n = normalized (roll=0) | r = raw camera | '
                'head: Euler xyz, n == r on pitch/yaw | '
                'wld: world head (elev-30, azim) | HCS: invariant',
                (12, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
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
            put(f"wld  ({r['world'][0]:+5.1f},{r['world'][1]:+5.1f})", 96, (150, 60, 180))
            if r['camd'] is not None:
                put(f"camd {r['camd']:5.2f} deg", 116, (0, 120, 255))

    cv2.imwrite(str(OUT / 'all_datasets_normalized.png'), canvas)
    log.info(f'输出 {OUT / "all_datasets_normalized.png"} '
             f'({len(all_rows)} 数据集 × {N_SAMPLE} 张)')


if __name__ == '__main__':
    main()
