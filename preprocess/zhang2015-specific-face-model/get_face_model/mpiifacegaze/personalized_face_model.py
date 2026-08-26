"""MPIIFaceGaze 逐人个性化人脸模型（方案一移植：单相机多帧联合 BA）

与 xgaze 的差异（README）：每人单相机（实验室固定机位），无逐相机拆分；
内参逐人取原始数据 Calibration/Camera.mat（与 zhang2015-insightface 预处理同源）。

输入:
  - 特征点索引: /media/yanglinxuan/zyx/MPIIFaceGaze/landmarks/pXX.h5（facial_landmarks_2d，
    原图像素坐标，与 ylx 预处理产物同源逐行一致）
  - 内参: <raw>/pXX/Calibration/Camera.mat（cameraMatrix + distCoeffs 1x5）
输出 /media/yanglinxuan/ylx/mpiifacegaze_specific_face_model/face_models/pXX/ 下:
  - cam00_model6.txt / cam00_model28.txt（沿用 xgaze 的 {group}_ 命名，单人仅 cam00，
    下游按「组名+模型」查找的代码可跨数据集通用）
  - canonical_model28.txt（单人即 cam00 模型副本）/ summary.txt
建模指标留档 <本目录>/metrics/（含通用模型基线对比，同 xgaze）。
用法（仓库根目录运行；CPU 即可）:
  /ssd/conda/envs/yanglinxuan/opengaze/bin/python preprocess/zhang2015-specific-face-model/get_face_model/mpiifacegaze/personalized_face_model.py [-sb 0 -se 15] [--overwrite]
"""
import os
import sys
import argparse
import time

import cv2
import h5py
import numpy as np
import scipy.io as sio
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# face_model_core 在上级 get_face_model/；utils 在仓库根
sys.path.insert(0, os.path.abspath(os.path.join(PROJECT_ROOT, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(PROJECT_ROOT, '..', '..', '..', '..')))

import face_model_core as core                     # noqa: E402
from utils.logger import get_logger                # noqa: E402

log = get_logger('preprocess.specific_face_model.mpiifacegaze')

LM_DIR = '/media/yanglinxuan/zyx/MPIIFaceGaze/landmarks'
RAW_DIR = '/media/yanglinxuan/zyx/MPIIFaceGaze'
OUT_DIR = '/media/yanglinxuan/ylx/mpiifacegaze_specific_face_model/face_models'
METRICS_DIR = os.path.join(PROJECT_ROOT, 'metrics')
FACE_MODEL_FILE = os.path.join(PROJECT_ROOT, '..', '..', '..',
                               'zhang2015-insightface', 'face_model_xgaze.txt')
GEN_ROWS = [20, 23, 26, 29, 15, 19]                 # 通用 50 点模型中 GEN6 对应行
SUBJECTS = ['p{:02d}'.format(i) for i in range(15)]

N_TRAIN, N_TEST = 60, 60     # 每人采样帧数: 120 帧均匀采样, 奇偶分为 train/test


def process_subject(subject, gen6, overwrite=False):
    lm_path = os.path.join(LM_DIR, subject + '.h5')
    if not os.path.isfile(lm_path):
        log.info('  landmarks h5 不存在, 跳过: {}'.format(lm_path))
        return None
    mat = sio.loadmat(os.path.join(RAW_DIR, subject, 'Calibration', 'Camera.mat'))
    K = np.asarray(mat['cameraMatrix'], dtype=float)
    dist = np.asarray(mat['distCoeffs'], dtype=float).reshape(1, 5)
    f_px = core.f_px_of(K)

    with h5py.File(lm_path, 'r') as f:
        lm_all = f['facial_landmarks_2d'][:]
    n_rows = len(lm_all)
    if n_rows < 20:
        log.warning('  仅 {} 帧观测, 跳过'.format(n_rows))
        return None

    # 采样 120 帧（全序列均匀），奇偶分 train/test
    n_sample = min(N_TRAIN + N_TEST, n_rows)
    sel = np.linspace(0, n_rows - 1, n_sample).astype(int)
    train_rows, test_rows = sorted(sel[0::2].tolist()), sorted(sel[1::2].tolist())

    # 单相机组：初始化 + 视角过滤
    def build_views(rows):
        views, angles = [], []
        for r in rows:
            lm_n, pose, ang = core.init_view(lm_all[r].astype(np.float64), K, dist, gen6)
            views.append((lm_n, pose))
            angles.append(ang)
        return views, angles

    train_views, train_angles = build_views(train_rows)
    test_views, test_angles = build_views(test_rows)
    train_views = [v for v, a in zip(train_views, train_angles) if a < core.VIEW_ANGLE_MAX]
    test_views = [v for v, a in zip(test_views, test_angles) if a < core.VIEW_ANGLE_MAX]
    med_angle = float(np.median(train_angles))
    log.info('  单相机 | 中位姿态角 {:.0f}° | train {} → {} 视图(<{:.0f}°) | test {} 视图'.format(
        med_angle, len(train_rows), len(train_views), core.VIEW_ANGLE_MAX, len(test_views)))
    if med_angle >= core.CAM_ANGLE_MAX:
        log.warning('  中位姿态角 {:.0f}° ≥ {:.0f}°（实验室机位偏正，一般不会触发；仍建模）'.format(
            med_angle, core.CAM_ANGLE_MAX))
    if len(train_views) < core.MIN_GROUP_VIEWS:
        log.warning('  可用训练视图 {} < {}（仍建模，标记偏少）'.format(
            len(train_views), core.MIN_GROUP_VIEWS))
    if not train_views or not test_views:
        return None

    # 建模 + 留出诊断（含通用模型基线）
    g = core.model_group(train_views, f_px)
    idx6_rows = [core.RIGID.index(i) for i in core.IDX6]
    test_rms, test_gen = core.eval_group(g['model'], test_views, f_px, gen6, idx6_rows)
    iod = float(np.linalg.norm(g['model'][core.RIGID.index(35)]
                               - g['model'][core.RIGID.index(93)]))

    # 保存（单组：cam00；canonical 即其副本）
    sub_dir = os.path.join(OUT_DIR, subject)
    ok = core.save_models(
        sub_dir, {'cam00': g['model']}, g['model'], idx6_rows,
        ['cam00  {:.2f}  {:.2f}  {:.1f}\n'.format(g['train_rms'], test_rms, iod)],
        overwrite=overwrite)
    log.info('  train RMS {:.2f} px | test RMS {:.2f} px | 通用基线 {:.2f} px '
             '({:.1f}x) | IOD {:.1f} mm -> {}'.format(
                 g['train_rms'], test_rms, test_gen,
                 test_gen / max(test_rms, 1e-6), iod, sub_dir))

    # 指标留档
    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(os.path.join(METRICS_DIR, subject + '.txt'), 'w') as f:
        f.write('# 组数 1 | train RMS {:.2f} px | test RMS {:.2f} px | 通用模型 test RMS '
                '{:.2f} px | 改善 {:.1f}x | IOD {:.1f} mm\n'.format(
                    g['train_rms'], test_rms, test_gen,
                    test_gen / max(test_rms, 1e-6), iod))
        f.write('# group  n_train_kept  train_rms_px  test_rms_px  '
                'test_rms_generic_px  iod_mm\n')
        f.write('cam00  {}  {:.2f}  {:.2f}  {:.2f}  {:.1f}\n'.format(
            g['n_kept'], g['train_rms'], test_rms, test_gen, iod))
    return {'subject': subject, 'n_groups': 1, 'train_med': g['train_rms'],
            'test_med': test_rms, 'test_gen_med': test_gen,
            'imp': test_gen / max(test_rms, 1e-6), 'iod_mean': iod,
            'saved': ok}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MPIIFaceGaze 逐人个性化人脸建模')
    parser.add_argument('-sb', '--subject_begin', type=int, help='起始被试编号(含，0~14)')
    parser.add_argument('-se', '--subject_end', type=int, help='结束被试编号(不含)')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已存在的输出')
    args = parser.parse_args()

    gen6 = np.loadtxt(FACE_MODEL_FILE)[GEN_ROWS, :]
    sb = args.subject_begin if args.subject_begin is not None else 0
    se = args.subject_end if args.subject_end is not None else len(SUBJECTS)

    rows = []
    for subject in SUBJECTS[sb:se]:
        if os.path.exists(os.path.join(OUT_DIR, subject, 'summary.txt')) and not args.overwrite:
            log.info('{}: 已存在, 跳过'.format(subject))
            continue
        t0 = time.time()
        log.info('{}:'.format(subject))
        row = process_subject(subject, gen6, overwrite=args.overwrite)
        if row is not None:
            rows.append(row)
        log.info('  用时 {:.1f}s'.format(time.time() - t0))

    os.makedirs(METRICS_DIR, exist_ok=True)
    csv_path = os.path.join(METRICS_DIR, 'summary_all.csv')
    with open(csv_path, 'w') as f:
        f.write('subject,n_groups,train_rms_med_px,test_rms_med_px,'
                'test_rms_generic_med_px,improvement_x,iod_mean_mm\n')
        for r in rows:
            f.write('{subject},{n_groups},{train_med:.2f},{test_med:.2f},'
                    '{test_gen_med:.2f},{imp:.1f},{iod_mean:.1f}\n'.format(**r))
    if rows:
        log.info('完成: {} 人 | train 中位 {:.2f} px | test 中位 {:.2f} px | 通用基线中位 '
                 '{:.2f} px | 改善中位 {:.1f}x | 指标留档 {}'.format(
                     len(rows),
                     np.median([r['train_med'] for r in rows]),
                     np.median([r['test_med'] for r in rows]),
                     np.median([r['test_gen_med'] for r in rows]),
                     np.median([r['imp'] for r in rows]), csv_path))
    else:
        log.info('完成: 0 人（全部跳过或无可用数据）')
