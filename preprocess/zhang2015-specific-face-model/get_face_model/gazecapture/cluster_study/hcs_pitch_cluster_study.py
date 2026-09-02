"""GC HCS pitch>0 簇归因研究：是否 orientation==2（2026-08-30）

背景：gaze_distribution_specific.png 中 GC 的 HCS 分布在 pitch>0、yaw≈0 处
存在一个簇（主体在 pitch≈−40）。本脚本与分布图同链路（gen_xe6 6 点 PnP +
PoG 官方链 + 标准系归一化）全量重采样 GC，逐样本记录 orientation/设备/角度，
统计簇成员的 orientation 构成，并导出簇内典型样本的原始帧与标注上下文。

判定：若簇内样本 orientation 几乎全为 2（或某单一朝向），则簇=该朝向的系统
行为；若 orientation 混合，则另查（设备/会话/标注）。

输出（本目录）:
  hcs_pitch_cluster_data.csv     逐样本记录（session/ori/device/帧/CCS/HCS）
  cluster_summary.txt            簇 × 朝向 交叉表 + 各朝向统计
  cluster_samples/               簇内典型样本原始帧 jpg + context.json
用法（仓库根目录）:
  /ssd/conda/envs/yanglinxuan/opengaze/bin/python \
  preprocess/zhang2015-specific-face-model/get_face_model/gazecapture/hcs_pitch_cluster_study.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'preprocess/zhang2015-specific-face-model/get_face_model'))

import face_model_core as core                              # noqa: E402
from utils.logger import get_logger                         # noqa: E402
from utils.normalization import (estimateHeadPose, normalizeData_face,  # noqa: E402
                                 vector_to_angles)

# 复用分布图脚本的 GEN_XE6 与 GC 官方链（唯一实现，避免手抄漂移）
_spec = importlib.util.spec_from_file_location(
    'sample_all',
    PROJECT_ROOT / 'preprocess/zhang2015-specific-face-model/viz/sample_all_datasets.py')
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)
_gc_spec = importlib.util.spec_from_file_location(
    'gc_pre', PROJECT_ROOT / 'preprocess/zhang2015-insightface/gazecapture/preprocessor.py')
gc_pre = importlib.util.module_from_spec(_gc_spec)
_gc_spec.loader.exec_module(gc_pre)

log = get_logger('preprocess.specific_face_model.gc.hcs_cluster')

LM = Path('/media/yanglinxuan/zyx/GazeCapture/landmarks')
RAW = Path('/media/yanglinxuan/zyx/GazeCapture')
CAL = Path('/media/yanglinxuan/zyx/GazeCapture/calibration')
N_FRAMES_PER_SESS = 10
CLUSTER_PITCH_MIN = 5.0        # 簇定义：HCS pitch > +5°
CLUSTER_YAW_ABS = 10.0         #          且 |yaw| < 10°
N_EXPORT = 8                   # 导出典型样本数


def main():
    sess = [(sp, p.stem) for sp in ('train', 'test')
            for p in sorted(LM.joinpath(sp).glob('*.h5'))]
    rows = []
    DUMMY = np.zeros((32, 32, 3), np.uint8)
    for sp, s in tqdm(sess, desc='sessions', unit='sess'):
        rec = RAW / s
        try:
            device = json.load(open(rec / 'info.json'))['DeviceName']
            dot = json.load(open(rec / 'dotInfo.json'))
            pos = {int(n.split('.')[0]): i for i, n in
                   enumerate(json.load(open(rec / 'frames.json')))}
        except Exception:
            continue
        slug = device.lower().replace(' ', '-')
        cals = {}
        for (w, h) in ((480, 640), (640, 480)):
            cal = CAL / f'{slug}_{w}x{h}.xml'
            if cal.is_file():
                fs = cv2.FileStorage(str(cal), cv2.FILE_STORAGE_READ)
                cals[(w, h)] = (fs.getNode('Camera_Matrix').mat(),
                                fs.getNode('Distortion_Coefficients').mat())
                fs.release()
        if not cals:
            continue
        with h5py.File(LM / sp / f'{s}.h5', 'r') as f:
            fr = f['frame_index'][:].ravel()
            ori_arr = f['orientation'][:].ravel()
            lm_all = f['facial_landmarks_2d'][:]
        rng = np.random.default_rng(abs(hash(s)) % (2 ** 32))
        idx = rng.choice(len(fr), size=min(N_FRAMES_PER_SESS, len(fr)),
                         replace=False)
        for i in idx:
            fidx, o, lm106 = int(fr[i]), int(ori_arr[i]), lm_all[i]
            pi = pos.get(fidx)
            if pi is None or dot['DotNum'][pi] == -1:
                continue
            w, h = (480, 640) if o in (1, 2) else (640, 480)
            if (w, h) not in cals:
                continue
            K, dist = cals[(w, h)]
            xc, yc = dot['XCam'][pi], dot['YCam'][pi]
            ccs_x, ccs_y = gc_pre._dot_to_ccs_mm(o, xc, yc)
            if ccs_y <= 0:
                continue                     # invalid_dot（官方门）
            gp = np.array(gc_pre._gaze_point_cam(o, ccs_x, ccs_y)).reshape(3, 1)
            try:
                rvec, tvec = estimateHeadPose(
                    lm106[m.IDX6].reshape(6, 1, 2).astype(float),
                    m.GEN_XE6, K, dist)
            except cv2.error:
                continue
            _, hr, gc = normalizeData_face(
                DUMMY, m.GEN_XE6, rvec, tvec, gp, K,
                fixed_forward=False)[:3]
            hR = cv2.Rodrigues(hr)[0]
            hcs = np.degrees(vector_to_angles((hR.T @ gc).ravel()))
            ccs = np.degrees(vector_to_angles(gc.ravel()))
            rows.append((s, sp, fidx, o, device, ccs[0], ccs[1],
                         hcs[0], hcs[1]))

    import csv as _csv
    csv_path = HERE / 'hcs_pitch_cluster_data.csv'
    with open(csv_path, 'w', newline='') as f:
        w_ = _csv.writer(f)
        w_.writerow(['session', 'split', 'frame', 'orientation', 'device',
                     'ccs_pitch', 'ccs_yaw', 'hcs_pitch', 'hcs_yaw'])
        w_.writerows(rows)

    arr = np.array([r[7] for r in rows])         # hcs_pitch
    yaw = np.array([r[8] for r in rows])
    ori = np.array([r[3] for r in rows])
    in_cluster = (arr > CLUSTER_PITCH_MIN) & (np.abs(yaw) < CLUSTER_YAW_ABS)
    lines = [f'样本总数 {len(rows):,}；簇定义: HCS pitch>{CLUSTER_PITCH_MIN:g}° 且 '
             f'|yaw|<{CLUSTER_YAW_ABS:g}°；簇内 {in_cluster.sum():,} '
             f'({in_cluster.mean() * 100:.2f}%)', '']
    lines.append('簇内 orientation 构成:')
    for o in (1, 2, 3, 4):
        n = int(((ori == o) & in_cluster).sum())
        lines.append(f'  ori{o}: {n:5d} '
                     f'({n / max(in_cluster.sum(), 1) * 100:5.1f}% of 簇; '
                     f'{n / max((ori == o).sum(), 1) * 100:5.1f}% of ori{o} 全体)')
    lines.append('')
    lines.append('各 orientation 的 HCS pitch 分布（全体）:')
    for o in (1, 2, 3, 4):
        v = arr[ori == o]
        if len(v):
            lines.append(f'  ori{o}: n={len(v):6d}  中位 {np.median(v):+7.2f}°  '
                         f'p10 {np.percentile(v, 10):+7.2f}  '
                         f'p90 {np.percentile(v, 90):+7.2f}  '
                         f'pitch>+5° 占 {(v > 5).mean() * 100:5.1f}%')
    lines.append('')
    sess_cl = sorted({r[0] for r, c in zip(rows, in_cluster) if c})
    lines.append(f'簇涉及的 session 数: {len(sess_cl)}'
                 f'（示例: {sess_cl[:8]}）')
    dev_cl = {}
    for r, c in zip(rows, in_cluster):
        if c:
            dev_cl[r[4]] = dev_cl.get(r[4], 0) + 1
    lines.append(f'簇内设备构成: {dev_cl}')
    summary = '\n'.join(lines)
    (HERE / 'cluster_summary.txt').write_text(summary + '\n')
    log.info('\n' + summary)

    # ---- 导出簇内典型样本（原始帧 + 上下文）----
    out_dir = HERE / 'cluster_samples'
    out_dir.mkdir(exist_ok=True)
    cand = [r for r, c in zip(rows, in_cluster) if c]
    cand.sort(key=lambda r: -r[7])               # pitch 最大在前
    seen_sess = set()
    exported = 0
    for r in cand:
        if r[0] in seen_sess or exported >= N_EXPORT:
            continue
        seen_sess.add(r[0])
        s, sp, fidx, o, device = r[0], r[1], r[2], r[3], r[4]
        img = cv2.imread(str(RAW / s / 'frames' / f'{fidx:05d}.jpg'))
        if img is None:
            continue
        tag = f'{s}_f{fidx:05d}_o{o}'
        cv2.imwrite(str(out_dir / f'{tag}.jpg'), img)
        dot = json.load(open(RAW / s / 'dotInfo.json'))
        pi = {int(n.split(".")[0]): i for i, n in
              enumerate(json.load(open(RAW / s / 'frames.json')))}[fidx]
        ctx = {'session': s, 'frame': fidx, 'orientation': o,
               'device': device, 'hcs_pitch': r[7], 'hcs_yaw': r[8],
               'ccs_pitch': r[5], 'ccs_yaw': r[6],
               'XCam': dot['XCam'][pi], 'YCam': dot['YCam'][pi],
               'DotNum': dot['DotNum'][pi]}
        (out_dir / f'{tag}_context.json').write_text(
            json.dumps(ctx, indent=2))
        exported += 1
    log.info(f'输出 {csv_path.name} / cluster_summary.txt / '
             f'cluster_samples/（{exported} 例）')


if __name__ == '__main__':
    main()
