"""EVE 逐被试个性化人脸模型（方案一移植：单相机多帧联合 BA，逐相机交付）

与 xgaze 的差异（README）：每刺激步 4 相机（basler 高质量 + 3 webcam），
内参逐相机取原始数据 <subject>/<step>/<cam>.h5 的 camera_matrix（实测同一被试
同一相机的 K 跨 step 恒定；mp4 官方已去畸变 → 畸变系数取零）。
观测取自 landmarks 索引 h5（facial_landmarks_2d + cam_index，5Hz 采样），
「相机组」= cam_index（attrs['cameras'] 顺序：basler/webcam_l/webcam_c/webcam_r），
组选择沿用 xgaze 准则（中位姿态角 <40° 且训练视图 ≥15）——侧视 webcam 会被自然排除。

输入:
  - 特征点索引: /media/yanglinxuan/zyx/EVE_dataset/eve_dataset/landmarks/<split>/<被试>.h5
  - 内参: <raw>/<被试>/<任一 step>/<cameras[c]>.h5 的 camera_matrix
输出 /media/yanglinxuan/ylx/eve_specific_face_model/face_models/<被试>/ 下:
  - cam{c}_model6.txt / cam{c}_model28.txt（c 与预处理 h5 的 cam_index 对齐）
  - canonical_model28.txt / summary.txt
建模指标留档 <本目录>/metrics/。
用法（仓库根目录运行；CPU 即可）:
  .../personalized_face_model.py [-sb 0 -se 44] [--overwrite]
"""
import os
import sys
import argparse
import time

import cv2
import h5py
import numpy as np
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(PROJECT_ROOT, '..')))
sys.path.insert(0, os.path.abspath(os.path.join(PROJECT_ROOT, '..', '..', '..', '..')))

import face_model_core as core                     # noqa: E402
from utils.logger import get_logger                # noqa: E402

log = get_logger('preprocess.specific_face_model.eve')

LM_ROOT = '/media/yanglinxuan/zyx/EVE_dataset/eve_dataset/landmarks'
RAW_DIR = '/media/yanglinxuan/zyx/EVE_dataset/eve_dataset'
OUT_DIR = '/media/yanglinxuan/ylx/eve_specific_face_model/face_models'
METRICS_DIR = os.path.join(PROJECT_ROOT, 'metrics')
FACE_MODEL_FILE = os.path.join(PROJECT_ROOT, '..', '..', '..',
                               'zhang2015-insightface', 'face_model_xgaze.txt')
GEN_ROWS = [20, 23, 26, 29, 15, 19]

N_TRAIN, N_TEST = 60, 60     # 每相机采样帧数: 120 帧均匀采样, 奇偶分 train/test


def load_K(subject, cam_name, steps):
    """逐相机内参：取该被试第一个在盘的 step 的 <cam>.h5 camera_matrix（跨 step 恒定）"""
    for step in steps:
        p = Path(RAW_DIR) / subject / step / (cam_name + '.h5')
        if p.is_file():
            with h5py.File(p, 'r') as f:
                return np.array(f['camera_matrix'], dtype=float)
    raise FileNotFoundError('{} 下未找到任何 {}/{}.h5'.format(RAW_DIR, subject, cam_name))


def process_subject(split, subject, gen6, overwrite=False):
    lm_path = os.path.join(LM_ROOT, split, subject + '.h5')
    if not os.path.isfile(lm_path):
        log.info('  landmarks h5 不存在, 跳过: {}'.format(lm_path))
        return None
    import json
    with h5py.File(lm_path, 'r') as f:
        cam_all = f['cam_index'][:].ravel()
        lm_all = f['facial_landmarks_2d'][:]
        cameras = json.loads(f.attrs['cameras'])     # ["basler","webcam_l","webcam_c","webcam_r"]
        steps = json.loads(f.attrs['steps'])

    dist = np.zeros((1, 5), dtype=float)   # mp4 官方已去畸变
    idx6_rows = [core.RIGID.index(i) for i in core.IDX6]

    models, train_rms, test_rms, test_gen, iods, n_kept = {}, {}, {}, {}, {}, {}
    for c, cam_name in enumerate(cameras):
        K = load_K(subject, cam_name, steps)
        f_px = core.f_px_of(K)
        rows_c = np.where(cam_all == c)[0]
        if len(rows_c) < 20:
            log.info('  cam{}({}) 仅 {} 帧观测, 跳过该相机'.format(c, cam_name, len(rows_c)))
            continue

        # 该相机采样 120 帧，奇偶分 train/test
        n_sample = min(N_TRAIN + N_TEST, len(rows_c))
        sel = rows_c[np.linspace(0, len(rows_c) - 1, n_sample).astype(int)]
        train_rows, test_rows = sorted(sel[0::2].tolist()), sorted(sel[1::2].tolist())

        def build_views(rows):
            vs, angles = [], []
            for r in rows:
                lm_n, pose, ang = core.init_view(lm_all[r].astype(np.float64), K, dist, gen6)
                vs.append((lm_n, pose))
                angles.append(ang)
            return ([v for v, a in zip(vs, angles) if a < core.VIEW_ANGLE_MAX], angles)

        train_views, angles = build_views(train_rows)
        test_views, _ = build_views(test_rows)
        med = float(np.median(angles)) if angles else 180.0
        n_views = len(train_views)
        if med >= core.CAM_ANGLE_MAX or n_views < core.MIN_GROUP_VIEWS:
            log.info('  cam{}({}) 排除: 中位角 {:.0f}°, 训练视图 {}'.format(
                c, cam_name, med, n_views))
            continue
        if not test_views:
            log.info('  cam{}({}) 无留出视图, 跳过'.format(c, cam_name))
            continue

        res = core.model_group(train_views, f_px)
        te, te_gen = core.eval_group(res['model'], test_views, f_px, gen6, idx6_rows)
        gname = 'cam{:02d}'.format(c)
        models[gname] = res['model']
        train_rms[gname], test_rms[gname], test_gen[gname] = res['train_rms'], te, te_gen
        n_kept[gname] = res['n_kept']
        iods[gname] = float(np.linalg.norm(res['model'][core.RIGID.index(35)]
                                           - res['model'][core.RIGID.index(93)]))
        log.info('  {}({}) train RMS {:.2f} px | test RMS {:.2f} px | 通用基线 {:.2f} px'.format(
            gname, cam_name, res['train_rms'], te, te_gen))

    if not models:
        log.warning('  无可用相机组, 跳过')
        return None
    if len(models) < 4:
        log.info('  仅 {}/4 个相机组通过选择准则（basler 正视通常保留，侧视 webcam 常被 '
                 '中位角准则排除）'.format(len(models)))

    canonical = core.canonical_of(models)
    sub_dir = os.path.join(OUT_DIR, subject)
    ok = core.save_models(
        sub_dir, models, canonical, idx6_rows,
        ['{}  {:.2f}  {:.2f}  {:.1f}\n'.format(g, train_rms[g], test_rms[g], iods[g])
         for g in models],
        overwrite=overwrite)

    med_tr = float(np.median(list(train_rms.values())))
    med_te = float(np.nanmedian(list(test_rms.values())))
    med_ge = float(np.nanmedian(list(test_gen.values())))
    mean_iod = float(np.mean(list(iods.values())))
    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(os.path.join(METRICS_DIR, subject + '.txt'), 'w') as f:
        f.write('# 相机组 {} | train RMS 中位 {:.2f} px | test RMS 中位 {:.2f} px | '
                '通用模型 test RMS 中位 {:.2f} px | 改善 {:.1f}x | IOD 均值 {:.1f} mm\n'.format(
                    sorted(models), med_tr, med_te, med_ge,
                    med_ge / max(med_te, 1e-6), mean_iod))
        f.write('# group  n_train_kept  train_rms_px  test_rms_px  '
                'test_rms_generic_px  iod_mm\n')
        for g in models:
            f.write('{}  {}  {:.2f}  {:.2f}  {:.2f}  {:.1f}\n'.format(
                g, n_kept[g], train_rms[g], test_rms[g], test_gen[g], iods[g]))
    log.info('  建模 {} 相机 | train RMS 中位 {:.2f} px | test RMS 中位 {:.2f} px | '
             '通用基线 {:.2f} px ({:.1f}x) | IOD {:.1f} mm -> {}'.format(
                 len(models), med_tr, med_te, med_ge,
                 med_ge / max(med_te, 1e-6), mean_iod, sub_dir))
    return {'subject': subject, 'n_groups': len(models), 'train_med': med_tr,
            'test_med': med_te, 'test_gen_med': med_ge,
            'imp': med_ge / max(med_te, 1e-6), 'iod_mean': mean_iod}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EVE 逐被试个性化人脸建模')
    parser.add_argument('-sb', '--subject_begin', type=int, help='起始被试编号(含)')
    parser.add_argument('-se', '--subject_end', type=int, help='结束被试编号(不含)')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已存在的输出')
    args = parser.parse_args()

    gen6 = np.loadtxt(FACE_MODEL_FILE)[GEN_ROWS, :]
    subjects = [(sp, p.stem) for sp in ('train', 'test')
                for p in sorted(Path(LM_ROOT, sp).glob('*.h5'))]
    sb = args.subject_begin if args.subject_begin is not None else 0
    se = args.subject_end if args.subject_end is not None else len(subjects)

    rows = []
    for split, subject in subjects[sb:se]:
        if os.path.exists(os.path.join(OUT_DIR, subject, 'summary.txt')) and not args.overwrite:
            log.info('{}: 已存在, 跳过'.format(subject))
            continue
        t0 = time.time()
        log.info('{}:'.format(subject))
        row = process_subject(split, subject, gen6, overwrite=args.overwrite)
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
        log.info('完成: {} 被试 | train 中位 {:.2f} px | test 中位 {:.2f} px | 通用基线中位 '
                 '{:.2f} px | 改善中位 {:.1f}x | 指标留档 {}'.format(
                     len(rows),
                     np.median([r['train_med'] for r in rows]),
                     np.median([r['test_med'] for r in rows]),
                     np.median([r['test_gen_med'] for r in rows]),
                     np.median([r['imp'] for r in rows]), csv_path))
    else:
        log.info('完成: 0 被试（全部跳过或无可用数据）')
