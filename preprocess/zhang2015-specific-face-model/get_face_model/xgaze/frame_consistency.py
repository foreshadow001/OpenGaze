"""ETH-XGaze 跨相机世界坐标系一致性检测（2026-08-28，使用全部 18 台相机）

与 EVE 版（../eve/frame_consistency.py）同协议，差异：
  - 相机集 = 全部 18 台中「可建模」的：逐视图 <60° 过滤后训练视图 ≥15
    （不施加中位角 <40° 的相机选择——按用户要求纳入极端视角相机；
    少数被试的 cam13 等因可用视图不足被跳过并留档）
  - 帧宇宙跨相机共享（同一 frame_index 同步拍摄），120 帧采样奇偶分 train/test，
    建模只用 train 的 60 帧
  - 额外按「相机对所属类别」分组汇总：双选定(<40°) / 混合 / 双非选定(≥40°)，
    观察不一致是否随视角极端程度增长

两问题：
  Q1（S 组）：逐相机 S 模型（三角化→剔帧→BA）6 关键点，名义同世界系（通用脸系）→ 重合？误差？
  Q2（三角化）：逐相机纯三角化模型（姿态=GEN6 PnP 固定，零优化自由度）→ 重合？误差？

误差定义：raw=直接逐点距离；centroid=质心距离（平移分量）；aligned=6 点 Kabsch
最优刚体对齐后逐点距离（纯形状差异）。

输出: metrics/frame_consistency/{per_pair.csv, per_point.csv, aggregate.md}
用法（仓库根目录）: .../frame_consistency.py [-j 14]
"""
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys
import argparse
from pathlib import Path
from multiprocessing import Pool

import cv2
import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE.parent))                      # get_face_model/
sys.path.insert(0, str(PROJECT_ROOT))

import face_model_core as core                           # noqa: E402
from utils.logger import get_logger                      # noqa: E402

log = get_logger('preprocess.specific_face_model.xgaze.frame_consistency')

LM_DIR = '/media/yanglinxuan/ylx/xgaze_insightface_224'
CALIB_DIR = '/media/yanglinxuan/Expansion/xgaze_raw/calibration/cam_calibration'
METRICS_DIR = HERE / 'metrics' / 'frame_consistency'
GEN6 = None
KS = DISTS = None
MIN_VIEWS = 15
IDX6_LABELS = ['eye35(外角)', 'eye39(内角)', 'eye89(内角)',
               'eye93(外角)', 'nose78(鼻底)', 'nose84(鼻底)']


def init_worker():
    global GEN6, KS, DISTS
    GEN6 = np.loadtxt(PROJECT_ROOT / 'preprocess/zhang2015-insightface/face_model_xgaze.txt')[
        [20, 23, 26, 29, 15, 19], :]
    KS, DISTS = {}, {}
    for c in range(18):
        fs = cv2.FileStorage(str(Path(CALIB_DIR) / f'cam{c:02d}.xml'),
                             cv2.FILE_STORAGE_READ)
        KS[c] = fs.getNode('Camera_Matrix').mat()
        DISTS[c] = fs.getNode('Distortion_Coefficients').mat()
        fs.release()


def process_subject(sid):
    subject = f'subject{sid:04d}'
    idx6_rows = [core.RIGID.index(i) for i in core.IDX6]
    with h5py.File(Path(LM_DIR) / f'{subject}.h5', 'r') as f:
        fr_all = f['frame_index'][:].ravel()
        cam_all = f['cam_index'][:].ravel()
        lm_all = f['facial_landmarks_2d'][:]

    frames_universe = sorted(set(fr_all.tolist()))
    sel = np.array(frames_universe)[np.linspace(
        0, len(frames_universe) - 1, min(120, len(frames_universe))).astype(int)]
    train_frames = sorted(sel[0::2].tolist())

    models, med_angle, skipped = {}, {}, []
    for c in range(18):
        m = (cam_all == c) & np.isin(fr_all, train_frames)
        views, angles = [], []
        for r_, fr_ in zip(np.where(m)[0], fr_all[m]):
            lm_n, pose, ang = core.init_view(lm_all[r_].astype(np.float64),
                                             KS[c], DISTS[c], GEN6)
            if ang < core.VIEW_ANGLE_MAX:                 # 逐视图 <60° 过滤
                views.append((lm_n, pose))
                angles.append(ang)
        med_angle[c] = float(np.median(angles)) if angles else 180.0
        if len(views) < MIN_VIEWS:
            skipped.append((c, len(views)))
            continue
        lm_t = np.stack([v[0] for v in views])
        pv_t = np.stack([v[1] for v in views])
        T6 = core.triangulate(lm_t, pv_t)[idx6_rows]
        S6 = core.model_group(views, core.f_px_of(KS[c]))['model'][idx6_rows]
        models[c] = {'S': S6, 'T': T6}

    pair_rows, point_rows = [], []
    cams = sorted(models)
    for i, a in enumerate(cams):
        for b in cams[:i]:
            cat = ('both_sel' if med_angle[a] < 40 and med_angle[b] < 40 else
                   'both_unsel' if med_angle[a] >= 40 and med_angle[b] >= 40 else 'mixed')
            for arm in ('S', 'T'):
                A, B = models[a][arm], models[b][arm]
                raw = np.linalg.norm(A - B, axis=1)
                R, t = core.kabsch(A, B)
                ali = np.linalg.norm((R @ A.T).T + t - B, axis=1)
                pair_rows.append((subject, arm, f'cam{a:02d}', f'cam{b:02d}', cat,
                                  med_angle[a], med_angle[b],
                                  float(np.median(raw)),
                                  float(np.linalg.norm(A.mean(0) - B.mean(0))),
                                  float(np.median(ali))))
                for pi, lab in enumerate(IDX6_LABELS):
                    point_rows.append((subject, arm, f'cam{a:02d}', f'cam{b:02d}',
                                       cat, lab, float(raw[pi]), float(ali[pi])))
    return {'subject': subject, 'n_cams': len(models), 'skipped': skipped,
            'pairs': pair_rows, 'points': point_rows}


def main():
    parser = argparse.ArgumentParser(description='XGaze S/三角化模型跨相机世界系一致性（18 台）')
    parser.add_argument('-j', '--jobs', type=int, default=14)
    args = parser.parse_args()

    sids = sorted(int(p.stem.replace('subject', '')) for p in Path(LM_DIR).glob('subject*.h5'))
    log.info(f'一致性检测: {len(sids)} 被试 × 18 相机')
    pairs, points, ncams, skipped_all = [], [], [], {}
    with Pool(args.jobs, initializer=init_worker) as pool:
        for i, res in enumerate(pool.imap_unordered(process_subject, sids), 1):
            pairs.extend(res['pairs'])
            points.extend(res['points'])
            ncams.append(res['n_cams'])
            for c, n in res['skipped']:
                skipped_all[f'cam{c:02d}'] = skipped_all.get(f'cam{c:02d}', 0) + 1
    log.info(f'完成: 每被试可建模相机中位 {np.median(ncams):.0f} 台; '
             f'跳过统计 {skipped_all}; 共 {len(pairs)} 模型对')

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_DIR / 'per_pair.csv', 'w') as f:
        f.write('subject,arm,cam_a,cam_b,category,med_angle_a,med_angle_b,'
                'raw_med_mm,centroid_mm,aligned_med_mm\n')
        for r in pairs:
            f.write('{},{},{},{},{},{:.0f},{:.0f},{:.3f},{:.3f},{:.3f}\n'.format(*r))
    with open(METRICS_DIR / 'per_point.csv', 'w') as f:
        f.write('subject,arm,cam_a,cam_b,category,point,raw_mm,aligned_mm\n')
        for r in points:
            f.write('{},{},{},{},{},{},{:.3f},{:.3f}\n'.format(*r))

    lines = ['## 总体中位（mm）', '',
             '| 臂 | raw | centroid（平移） | Kabsch 对齐后 |', '|---|---|---|---|']
    for arm in ('S', 'T'):
        rs = [r for r in pairs if r[1] == arm]
        lines.append('| {} | {:.2f} | {:.2f} | {:.2f} |'.format(
            arm, np.median([x[7] for x in rs]), np.median([x[8] for x in rs]),
            np.median([x[9] for x in rs])))
    lines += ['', '## 按相机对类别（对齐后中位，mm）', '',
              '| 类别 | S raw | S aligned | T raw | T aligned | n对(每被试) |',
              '|---|---|---|---|---|---|']
    for cat, label in (('both_sel', '双选定(<40°)'), ('mixed', '混合'),
                       ('both_unsel', '双非选定(≥40°)')):
        vals = []
        for arm in ('S', 'T'):
            rs = [r for r in pairs if r[1] == arm and r[4] == cat]
            vals += [np.median([x[7] for x in rs]) if rs else float('nan'),
                     np.median([x[9] for x in rs]) if rs else float('nan')]
        n = len([r for r in pairs if r[4] == cat]) // 2
        lines.append(f'| {label} | {vals[0]:.2f} | {vals[1]:.2f} | '
                     f'{vals[2]:.2f} | {vals[3]:.2f} | {n} |')
    lines += ['', '## 逐关键点（对齐后中位，mm）', '', '| 臂 | 点 | raw | aligned |',
              '|---|---|---|---|']
    for arm in ('S', 'T'):
        for lab in IDX6_LABELS:
            rs = [r for r in points if r[1] == arm and r[5] == lab]
            lines.append('| {} | {} | {:.2f} | {:.2f} |'.format(
                arm, lab, np.median([x[6] for x in rs]),
                np.median([x[7] for x in rs])))
    (METRICS_DIR / 'aggregate.md').write_text('\n'.join(lines) + '\n')
    log.info(f'{METRICS_DIR / "aggregate.md"}\n' + '\n'.join(lines[:10]))


if __name__ == '__main__':
    main()
