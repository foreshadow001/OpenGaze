"""GazeCapture 逐 session 个性化人脸模型（方案一移植：单相机多帧联合 BA，逐朝向交付）

与 xgaze 的差异（README）：session 级身份（无跨 session 人 ID，逐 session 独立建模）、
单前置相机、内参为按设备分组的外部考证值（generate_calibration.py 生成的
calibration/<slug>_<w>x<h>.xml）而非官方逐相机标定。

「相机组」= 帧朝向（Orientation 1~4，同一 session 内混布）：竖屏(1/2)用 480x640、
横屏(3/4)用 640x480 内参（与 zhang2015-insightface 预处理逐帧查表同规则）；
各组独立 BA，组选择沿用 xgaze 准则（中位姿态角 <40° 且训练视图 ≥15）。

输入:
  - 特征点索引: /media/yanglinxuan/zyx/GazeCapture/landmarks/<split>/<session>.h5
    （facial_landmarks_2d + orientation，原图像素坐标）
  - 设备名: <raw>/<session>/info.json 的 DeviceName（→ calibration/<slug>_*.xml）
输出 /media/yanglinxuan/ylx/gazecapture_specific_face_model/face_models/<session>/ 下:
  - ori{o}_model6.txt / ori{o}_model28.txt（o=1..4，视角正常的朝向）
  - canonical_model28.txt / summary.txt
建模指标留档 <本目录>/metrics/。
用法（仓库根目录运行；CPU 即可）:
  .../personalized_face_model.py [--split train|test|all] [-sb 0 -se 100] [--overwrite]
"""
import os
import sys
import argparse
import json
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

log = get_logger('preprocess.specific_face_model.gazecapture')

LM_ROOT = '/media/yanglinxuan/zyx/GazeCapture/landmarks'
RAW_DIR = '/media/yanglinxuan/zyx/GazeCapture'
CALIB_DIR = '/media/yanglinxuan/zyx/GazeCapture/calibration'
OUT_DIR = '/media/yanglinxuan/ylx/gazecapture_specific_face_model/face_models'
METRICS_DIR = os.path.join(PROJECT_ROOT, 'metrics')
FACE_MODEL_FILE = os.path.join(PROJECT_ROOT, '..', '..', '..',
                               'zhang2015-insightface', 'face_model_xgaze.txt')
GEN_ROWS = [20, 23, 26, 29, 15, 19]
# ⚠️ 已弃用（2026-08-27）：个别设备/朝向的实际帧尺寸与此直觉相反（iPad Air 2
# 的 ori4 实为 480x640），内参查找一律按实际帧尺寸读 1 帧确定，勿再使用本表
ORI_SIZE = {1: (480, 640), 2: (480, 640), 3: (640, 480), 4: (640, 480)}

N_TRAIN, N_TEST = 60, 60     # 每 session 采样帧数: 120 帧均匀采样, 奇偶分 train/test
# 质量门槛（组级，防放宽角度后收进劣质模型）：留出 RMS 与 IOD 任一不合格即弃该组，
# 该 session 若无组合格则不产出（下游归一化回退通用模型）
TEST_RMS_MAX = 2.0           # px
IOD_RANGE = (80.0, 100.0)    # mm


def _group_ok(test_rms, iod):
    return (not np.isnan(test_rms)) and test_rms <= TEST_RMS_MAX \
        and IOD_RANGE[0] <= iod <= IOD_RANGE[1]


def _slugify(device_name):
    return device_name.lower().replace(' ', '-')


def _load_calib(device, w, h, cache):
    key = (device, w, h)
    if key not in cache:
        path = os.path.join(CALIB_DIR, '{}_{}x{}.xml'.format(_slugify(device), w, h))
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        K = fs.getNode('Camera_Matrix').mat()
        dist = fs.getNode('Distortion_Coefficients').mat()
        fs.release()
        cache[key] = (K, dist)
    return cache[key]


def process_session(split, session, gen6, calib_cache, max_angle=core.CAM_ANGLE_MAX,
                    overwrite=False):
    lm_path = os.path.join(LM_ROOT, split, session + '.h5')
    if not os.path.isfile(lm_path):
        log.info('  landmarks h5 不存在, 跳过: {}'.format(lm_path))
        return None
    try:
        device = json.load(open(os.path.join(RAW_DIR, session, 'info.json')))['DeviceName']
    except Exception as e:
        log.warning('  info.json 读取失败: {}'.format(e))
        return None

    with h5py.File(lm_path, 'r') as f:
        ori_all = f['orientation'][:].ravel()
        fr_all = f['frame_index'][:].ravel()
        lm_all = f['facial_landmarks_2d'][:]
    n_rows = len(lm_all)
    if n_rows < 20:
        log.warning('  仅 {} 帧观测, 跳过'.format(n_rows))
        return None

    # 采样 120 帧（全 session 均匀），奇偶分 train/test
    n_sample = min(N_TRAIN + N_TEST, n_rows)
    sel = np.linspace(0, n_rows - 1, n_sample).astype(int)
    train_rows, test_rows = sorted(sel[0::2].tolist()), sorted(sel[1::2].tolist())

    # 按朝向组建模观测。内参按该朝向【实际帧尺寸】查（读 1 帧定 w/h）——
    # app 存储帧 = 界面视角，个别设备/朝向的宽高与 ORI_SIZE 直觉相反
    # （如 iPad Air 2 的 ori4 实为 480x640），用错 K 会让 PnP 角度到 ~180°（垃圾）
    groups = {}
    for o in sorted(set(ori_all.tolist())):
        rows_o = [r for r in range(len(ori_all)) if int(ori_all[r]) == int(o)]
        img0 = cv2.imread(str(Path(RAW_DIR) / session / 'frames' /
                              '{:05d}.jpg'.format(int(fr_all[rows_o[0]]))))
        if img0 is None:
            log.warning('  ori{} 首帧读取失败, 跳过该朝向'.format(o))
            continue
        h_px, w_px = img0.shape[:2]
        K, dist = _load_calib(device, w_px, h_px, calib_cache)
        views = {}
        for tag, rows in (('train', train_rows), ('test', test_rows)):
            vs, angles = [], []
            for r in rows:
                if int(ori_all[r]) != int(o):
                    continue
                lm_n, pose, ang = core.init_view(lm_all[r].astype(np.float64), K, dist, gen6)
                vs.append((lm_n, pose))
                angles.append(ang)
            views[tag] = ([v for v, a in zip(vs, angles) if a < core.VIEW_ANGLE_MAX],
                          angles)
        groups[o] = {'K': K, 'f_px': core.f_px_of(K), 'views': views}

    # 朝向组选择：中位姿态角 <40° 且训练视图 ≥15（沿用 xgaze 相机选择准则）
    sel_groups = []
    for o, g in groups.items():
        train_views, angles = g['views']['train']
        n_views = len(train_views)
        med = float(np.median(angles)) if angles else 180.0
        if med < max_angle and n_views >= core.MIN_GROUP_VIEWS:
            sel_groups.append(o)
    log.info('  设备 {} | 朝向选择(中位角<{:.0f}°): {} | 各朝向中位角: {}'.format(
        device, max_angle, sel_groups,
        ' '.join('{}:{:.0f}°'.format(o, np.median(groups[o]['views']['train'][1])
                                     if groups[o]['views']['train'][1] else 180)
                 for o in sorted(groups))))
    if not sel_groups:
        log.warning('  无可用朝向组, 跳过')
        return None
    if len(sel_groups) < 2:
        log.info('  仅 {} 个朝向组（正常：手机使用常固定 1~3 种朝向）'.format(len(sel_groups)))

    models, train_rms, test_rms, test_gen, iods, n_kept = {}, {}, {}, {}, {}, {}
    idx6_rows = [core.RIGID.index(i) for i in core.IDX6]
    for o in sel_groups:
        g = groups[o]
        train_views = g['views']['train'][0]
        test_views = g['views']['test'][0]
        if not test_views:
            continue
        res = core.model_group(train_views, g['f_px'])
        te, te_gen = core.eval_group(res['model'], test_views, g['f_px'], gen6, idx6_rows)
        iod = float(np.linalg.norm(res['model'][core.RIGID.index(35)]
                                   - res['model'][core.RIGID.index(93)]))
        if not _group_ok(te, iod):
            log.info('  ori{} 质量门槛未过(test {:.2f}px / IOD {:.1f}mm), 弃'.format(
                o, te, iod))
            continue
        gname = 'ori{}'.format(o)          # 文件名键：ori{o}_model6/28.txt
        models[gname] = res['model']
        train_rms[gname], test_rms[gname], test_gen[gname] = res['train_rms'], te, te_gen
        n_kept[gname] = res['n_kept']
        iods[gname] = iod
    if not models:
        log.warning('  全部朝向组未过选择/质量门槛, 跳过（下游回退通用模型）')
        return None

    canonical = core.canonical_of(models)
    sub_dir = os.path.join(OUT_DIR, session)
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
    with open(os.path.join(METRICS_DIR, session + '.txt'), 'w') as f:
        f.write('# 设备 {} | 朝向组 {} | train RMS 中位 {:.2f} px | test RMS 中位 {:.2f} px | '
                '通用模型 test RMS 中位 {:.2f} px | 改善 {:.1f}x | IOD 均值 {:.1f} mm\n'.format(
                    device, sorted(models), med_tr, med_te, med_ge,
                    med_ge / max(med_te, 1e-6), mean_iod))
        f.write('# group  n_train_kept  train_rms_px  test_rms_px  '
                'test_rms_generic_px  iod_mm\n')
        for g in models:
            f.write('{}  {}  {:.2f}  {:.2f}  {:.2f}  {:.1f}\n'.format(
                g, n_kept[g], train_rms[g], test_rms[g], test_gen[g], iods[g]))
    log.info('  建模 {} 朝向 | train RMS 中位 {:.2f} px | test RMS 中位 {:.2f} px | '
             '通用基线 {:.2f} px ({:.1f}x) | IOD {:.1f} mm -> {}'.format(
                 len(models), med_tr, med_te, med_ge,
                 med_ge / max(med_te, 1e-6), mean_iod, sub_dir))
    return {'subject': session, 'n_groups': len(models), 'train_med': med_tr,
            'test_med': med_te, 'test_gen_med': med_ge,
            'imp': med_ge / max(med_te, 1e-6), 'iod_mean': mean_iod}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GazeCapture 逐 session 个性化人脸建模')
    parser.add_argument('--split', default='all', choices=['train', 'test', 'all'],
                        help='处理哪个 split 的 session（landmarks 目录下）')
    parser.add_argument('-sb', '--session_begin', type=int, help='起始序号(含，按排序后列表)')
    parser.add_argument('-se', '--session_end', type=int, help='结束序号(不含)')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已存在的输出')
    parser.add_argument('--max-angle', type=float, default=core.CAM_ANGLE_MAX,
                        help='朝向组选择的中位姿态角上限(默认 %.0f°)。补跑极端姿态 '
                             'session 用宽松值（如 65），组级质量门槛会兜底' % core.CAM_ANGLE_MAX)
    args = parser.parse_args()

    gen6 = np.loadtxt(FACE_MODEL_FILE)[GEN_ROWS, :]
    splits = ['train', 'test'] if args.split == 'all' else [args.split]
    sessions = []
    for sp in splits:
        sessions += [(sp, p.stem) for p in sorted(Path(LM_ROOT, sp).glob('*.h5'))]
    sb = args.session_begin if args.session_begin is not None else 0
    se = args.session_end if args.session_end is not None else len(sessions)

    calib_cache = {}
    rows = []
    for split, session in sessions[sb:se]:
        if os.path.exists(os.path.join(OUT_DIR, session, 'summary.txt')) and not args.overwrite:
            log.info('{}: 已存在, 跳过'.format(session))
            continue
        t0 = time.time()
        log.info('{}:'.format(session))
        row = process_session(split, session, gen6, calib_cache,
                              max_angle=args.max_angle, overwrite=args.overwrite)
        if row is not None:
            rows.append(row)
        log.info('  用时 {:.1f}s'.format(time.time() - t0))

    os.makedirs(METRICS_DIR, exist_ok=True)
    csv_path = os.path.join(METRICS_DIR, 'summary_all.csv')
    with open(csv_path, 'w') as f:
        f.write('session,n_groups,train_rms_med_px,test_rms_med_px,'
                'test_rms_generic_med_px,improvement_x,iod_mean_mm\n')
        for r in rows:
            f.write('{subject},{n_groups},{train_med:.2f},{test_med:.2f},'
                    '{test_gen_med:.2f},{imp:.1f},{iod_mean:.1f}\n'.format(**r))
    if rows:
        log.info('完成: {} session | train 中位 {:.2f} px | test 中位 {:.2f} px | 通用基线中位 '
                 '{:.2f} px | 改善中位 {:.1f}x | 指标留档 {}'.format(
                     len(rows),
                     np.median([r['train_med'] for r in rows]),
                     np.median([r['test_med'] for r in rows]),
                     np.median([r['test_gen_med'] for r in rows]),
                     np.median([r['imp'] for r in rows]), csv_path))
    else:
        log.info('完成: 0 session（全部跳过或无可用数据）')
