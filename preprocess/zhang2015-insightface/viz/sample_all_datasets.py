"""v1（zhang2015-insightface）四数据集各抽 20 张归一化可视化（2026-08-31）

与 v2 版（zhang2015-specific-face-model/viz/sample_all_datasets.py）样式与
数量一致：每数据集两行 20 张、块间分隔线、左侧粗体标签、每张带
id / head / gaze 标注。差异仅在链路：

- 直接读 v1 预处理产物 h5（face_patch + face_gaze + 三索引），不重算几何；
- head pose 由产物 h5 的 face_mat_norm（hR_norm）直接解算（is_true6=False
  旧口径——v1 用 gen6，不走标准系约定）；

输出: 本目录 all_datasets_normalized.png
用法（仓库根目录）:
  /ssd/conda/envs/yanglinxuan/opengaze/bin/python \
  preprocess/zhang2015-insightface/viz/sample_all_datasets.py
"""
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

_PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT))

from utils.logger import get_logger
from utils.normalization import vector_to_angles

log = get_logger('preprocess.insightface.viz.normalized_samples')

YLX = Path('/media/yanglinxuan/ylx')
HERE = Path(__file__).resolve().parent
N_SAMPLE = 20
PER_ROW = 10
W_IMG, H_IMG = 224, 224

DATASETS = [
    ('xgaze',         sorted(YLX.joinpath('xgaze_insightface_224').glob('subject*.h5'))),
    ('gazecapture',   sorted(YLX.joinpath('gazecapture_insightface_224').glob('*/*.h5'))),
    ('mpiifacegaze',  sorted(YLX.joinpath('mpiifacegaze_insightface_224').glob('p*.h5'))),
    ('eve',           sorted(YLX.joinpath('eve_insightface_224').glob('*/*.h5'))),
]


def load(name, files, rng):
    """每文件（被试/session）取 1 帧，不够 20 再补（与 v2 同策略）"""
    out, used = [], set()

    def one(fp):
        try:
            with h5py.File(fp, 'r') as f:
                n = f['face_patch'].shape[0]
                if n == 0:
                    return False
                i = int(rng.integers(n))
                if (str(fp), i) in used:
                    return False
                patch = f['face_patch'][i]
                gaze = np.array(f['face_gaze'][i])             # 弧度
                hR = np.array(f['face_mat_norm'][i]) if 'face_mat_norm' in f \
                    else np.eye(3)
                sid = fp.stem
                if name == 'gazecapture':
                    sid = f'{fp.parent.name}/{fp.stem}'
        except Exception:
            return False
        used.add((str(fp), i))
        out.append((patch, hR, np.degrees(gaze), f'{sid}'))
        return True

    for fp in files:
        if one(fp) and len(out) >= N_SAMPLE:
            break
    fail = 0
    while len(out) < N_SAMPLE and fail < 100:
        if not one(files[int(rng.integers(len(files)))]):
            fail += 1
    return out[:N_SAMPLE]


def main():
    rng = np.random.default_rng(42)
    all_rows = []
    for name, files in DATASETS:
        log.info(f'加载 {name}...')
        res = load(name, files, rng)
        log.info(f'{name}: {len(res)} 样本')
        all_rows.append((name, 'v1', res))

    # 出图（与 v2 版式一致）
    LABEL_W, PAD, BLOCK_GAP = 185, 6, 30
    CELL_H = H_IMG + 76
    N_ROWS = (N_SAMPLE + PER_ROW - 1) // PER_ROW
    W = LABEL_W + PER_ROW * (W_IMG + PAD) + PAD
    H = len(all_rows) * (N_ROWS * CELL_H + BLOCK_GAP) - BLOCK_GAP + PAD + 26
    canvas = np.full((H, W, 3), 255, np.uint8)
    cv2.putText(canvas, 'v1 pipeline (insightface_224, gen6) | '
                'head from face_mat_norm (legacy gen6 convention)',
                (12, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (60, 60, 60), 1, cv2.LINE_AA)

    TOP = 26
    fwd = np.array([0.0, 0.0, -1.0])
    for di, (name, tag, res) in enumerate(all_rows):
        y_top = TOP + PAD + di * (N_ROWS * CELL_H + BLOCK_GAP)
        img_h = (N_ROWS - 1) * CELL_H + W_IMG
        yc = y_top + img_h // 2
        if di > 0:
            cv2.line(canvas, (0, y_top - BLOCK_GAP // 2),
                     (W, y_top - BLOCK_GAP // 2), (200, 200, 200), 2)
        cv2.rectangle(canvas, (0, y_top),
                      (LABEL_W - 10, y_top + img_h - 1),
                      (232, 238, 246), -1)
        fs = 0.95
        while fs > 0.3 and cv2.getTextSize(
                name, cv2.FONT_HERSHEY_SIMPLEX, fs, 3)[0][0] > LABEL_W - 24:
            fs -= 0.05
        cv2.putText(canvas, name, (12, yc - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (25, 25, 110), 3, cv2.LINE_AA)
        cv2.putText(canvas, f'({tag})', (12, yc + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (90, 90, 90), 2, cv2.LINE_AA)
        for i, (patch, hR, gaze_deg, sid) in enumerate(res):
            rr, cc = divmod(i, PER_ROW)
            x0 = LABEL_W + cc * (W_IMG + PAD)
            y0 = y_top + rr * CELL_H
            canvas[y0:y0 + H_IMG, x0:x0 + W_IMG] = patch
            hp, hy = np.degrees(vector_to_angles(hR @ fwd))
            put = lambda txt, dy, c, dx=2: cv2.putText(
                canvas, txt, (x0 + dx, y0 + H_IMG + dy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1, cv2.LINE_AA)
            put(f'{sid[:17]}', 16, (90, 90, 90))
            put(f'head ({hp:+5.1f},{hy:+5.1f})', 36, (30, 30, 30))
            put(f'gaze ({gaze_deg[0]:+5.1f},{gaze_deg[1]:+5.1f})', 56, (0, 0, 200))

    out = HERE / 'all_datasets_normalized.png'
    cv2.imwrite(str(out), canvas)
    log.info(f'输出 {out}')


if __name__ == '__main__':
    main()
