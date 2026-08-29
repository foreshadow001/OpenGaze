"""EVE 跨相机世界坐标系一致性检测（2026-08-28）

两个问题（用户指定）：
  Q1（S 组）：逐相机 S 模型（三角化→剔帧→BA）的 6 关键点 3D 坐标，
     名义上都在同一个世界系（通用脸模型系，PnP 锚定）——直接比较是否重合？误差多少？
  Q2（三角化）：逐相机纯三角化模型（无剔帧无 BA，姿态=GEN6 PnP 输出，世界系严格锚定）
     的 6 关键点跨相机是否重合？误差多少？

  Q1 与 Q2 的对照回答「相机间不一致是 BA 造成的还是观测自带的」：
  T 模型没有任何优化自由度（姿态固定、纯线性求交），其跨相机差异完全来自观测本身。

误差报告（每个相机对、每个关键点）：
  raw      两模型直接逐点距离（名义同世界系，无任何对齐）——衡量"名义重合"程度
  centroid 两模型质心距离（raw 中的刚体平移分量）
  aligned  Kabsch 最优刚体对齐（仅旋转平移，不含缩放）后的逐点距离——纯形状差异
           （在 6 点上对齐，即"这 6 点在刚体变换下能 达到的最好重合"）

输入: landmarks 索引 h5 + 原始 step 内参；相机集 = 现行选择准则通过的相机（与 S 交付一致）
输出: metrics/frame_consistency/{per_pair.csv, per_point.csv, aggregate.md}
用法（仓库根目录）: .../frame_consistency.py [-j 12]
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys
import json
import argparse
import importlib.util
from pathlib import Path
from multiprocessing import Pool

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]                            # 仓库根
sys.path.insert(0, str(HERE.parent))                      # get_face_model/（face_model_core）
sys.path.insert(0, str(PROJECT_ROOT))

import face_model_core as core                           # noqa: E402
from utils.logger import get_logger                      # noqa: E402

_spec = importlib.util.spec_from_file_location('eve_study', HERE / 'multi_camera_study.py')
ms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ms)                             # 复用 load_K / build_views

log = get_logger('preprocess.specific_face_model.eve.frame_consistency')

LM_ROOT = '/media/yanglinxuan/zyx/EVE_dataset/eve_dataset/landmarks'
METRICS_DIR = HERE / 'metrics' / 'frame_consistency'
GEN6 = None
IDX6_LABELS = ['eye35(outer corner)', 'eye39(inner corner)', 'eye89(inner corner)',
               'eye93(outer corner)', 'nose78(base of nose)', 'nose84(base of nose)']


def init_worker():
    global GEN6
    GEN6 = np.loadtxt(PROJECT_ROOT / 'preprocess/zhang2015-insightface/face_model_xgaze.txt')[
        [20, 23, 26, 29, 15, 19], :]


def process_subject(args):
    split, subject = args
    idx6_rows = [core.RIGID.index(i) for i in core.IDX6]
    with h5py.File(Path(LM_ROOT) / split / f'{subject}.h5', 'r') as f:
        cam_all = f['cam_index'][:].ravel()
        lm_all = f['facial_landmarks_2d'][:]
        cameras = json.loads(f.attrs['cameras'])
        steps = json.loads(f.attrs['steps'])

    models = {}      # cam -> {'S': 6x3, 'T': 6x3}
    for c, cam_name in enumerate(cameras):
        K = ms.load_K(subject, cam_name, steps)
        rows_c = np.where(cam_all == c)[0]
        if len(rows_c) < 20:
            continue
        sel = rows_c[np.linspace(0, len(rows_c) - 1,
                                 min(120, len(rows_c))).astype(int)]
        train_rows, test_rows = sel[0::2], sel[1::2]
        train_views, angles = ms.build_views(lm_all, train_rows, K, GEN6)
        test_views, _ = ms.build_views(lm_all, test_rows, K, GEN6)
        med = float(np.median(angles)) if angles else 180.0
        if (med >= core.CAM_ANGLE_MAX or len(train_views) < core.MIN_GROUP_VIEWS
                or not test_views):
            continue
        lm_t = np.stack([v[0] for v in train_views])
        pv_t = np.stack([v[1] for v in train_views])
        T6 = core.triangulate(lm_t, pv_t)[idx6_rows]              # Q2: pure triangulation
        S6 = core.model_group(train_views, core.f_px_of(K))['model'][idx6_rows]  # Q1
        models[cam_name] = {'S': S6, 'T': T6}

    pair_rows, point_rows = [], []
    names = sorted(models)
    for i, a in enumerate(names):
        for b in names[:i]:
            for arm in ('S', 'T'):
                A, B = models[a][arm], models[b][arm]
                raw = np.linalg.norm(A - B, axis=1)
                R, t = core.kabsch(A, B)
                ali = np.linalg.norm((R @ A.T).T + t - B, axis=1)
                cen = float(np.linalg.norm(A.mean(0) - B.mean(0)))
                pair_rows.append((subject, arm, a, b, float(np.median(raw)),
                                  cen, float(np.median(ali))))
                for pi, lab in enumerate(IDX6_LABELS):
                    point_rows.append((subject, arm, a, b, lab,
                                       float(raw[pi]), float(ali[pi])))
    return {'subject': subject, 'n_cams': len(models), 'pairs': pair_rows,
            'points': point_rows}


def main():
    parser = argparse.ArgumentParser(description='EVE S/triangulated model cross-camera world-frame consistency')
    parser.add_argument('-j', '--jobs', type=int, default=12)
    args = parser.parse_args()

    subjects = [(sp, p.stem) for sp in ('train', 'test')
                for p in sorted(Path(LM_ROOT, sp).glob('*.h5'))]
    log.info(f'Consistency check: {len(subjects)} subjects')
    pairs, points, ncams = [], [], []
    with Pool(args.jobs, initializer=init_worker) as pool:
        for i, res in enumerate(pool.imap_unordered(
                process_subject, subjects), 1):
            pairs.extend(res['pairs'])
            points.extend(res['points'])
            ncams.append(res['n_cams'])
    log.info(f'Complete: median number of cameras per subject {np.median(ncams):.0f}, '
             f'{len(pairs)} model pairs')

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / 'per_pair.csv', 'w') as f:
        f.write('subject,arm,cam_a,cam_b,raw_med_mm,centroid_mm,aligned_med_mm\n')
        for r in pairs:
            f.write(f'{r[0]},{r[1]},{r[2]},{r[3]},{r[4]:.3f},{r[5]:.3f},{r[6]:.3f}\n')
    with open(METRICS_DIR / 'per_point.csv', 'w') as f:
        f.write('subject,arm,cam_a,cam_b,point,raw_mm,aligned_mm\n')
        for r in points:
            f.write(f'{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]:.3f},{r[6]:.3f}\n')

    lines = ['## Per camera-pair (44-subject medians, mm)', '',
             '| arm | cam pair | raw median | centroid distance | post-Kabsch-alignment median | n |',
             '|---|---|---|---|---|---|']
    agg = {}
    for r in pairs:
        agg.setdefault((r[1], f'{r[2]}↔{r[3]}'), []).append(r)
    for (arm, pair), rs in sorted(agg.items()):
        lines.append('| {} | {} | {:.2f} | {:.2f} | {:.2f} | {} |'.format(
            arm, pair, np.median([x[4] for x in rs]),
            np.median([x[5] for x in rs]), np.median([x[6] for x in rs]), len(rs)))
    lines += ['', '## Overall medians (mm)', '',
              '| arm | raw | centroid | aligned |', '|---|---|---|---|']
    for arm in ('S', 'T'):
        rs = [r for r in pairs if r[1] == arm]
        lines.append('| {} | {:.2f} | {:.2f} | {:.2f} |'.format(
            arm, np.median([x[4] for x in rs]), np.median([x[5] for x in rs]),
            np.median([x[6] for x in rs])))
    lines += ['', '## Per key point (aligned medians, mm)', '',
              '| arm | point | raw median | aligned median |', '|---|---|---|---|']
    for arm in ('S', 'T'):
        for lab in IDX6_LABELS:
            rs = [r for r in points if r[1] == arm and r[4] == lab]
            lines.append('| {} | {} | {:.2f} | {:.2f} |'.format(
                arm, lab, np.median([x[5] for x in rs]),
                np.median([x[6] for x in rs])))
    (METRICS_DIR / 'aggregate.md').write_text('\n'.join(lines) + '\n')
    log.info(f'{METRICS_DIR / "aggregate.md"}\n' + '\n'.join(lines[7:15]))


if __name__ == '__main__':
    main()
