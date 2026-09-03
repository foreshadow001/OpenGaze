"""EVE 侧面视角 zhang/ours 预处理对比可视化（2026-09-02 定稿）

一张 png = 两个子图（webcam_l / webcam_r，同一人同一时刻）上下叠放。每个子图：
  原始图像在左侧（高度 = 两个预处理图之和，**长宽比不变**）；
  预处理图像在右侧分两行——上=zhang（头姿归零）、下=ours（roll-only）；
  标注列在最右、与两行对齐。

标注内容（每行一个版本块）：
  来源 id；头姿（pre/post，欧拉 α,β）；视线 CCS（pre/post）；HCS
  （与归一化无关，两版本恒等；ours 的 head post≡pre 即 v3 不变量）。

数据缓存（原始图 + v2/v3 patch + 数值）到本目录 cache_side_view.npz，
缓存存在则直接出图；删除缓存或 CACHE_REFRESH=1 重新采样。
输出: 本目录 side_view_zhang_ours_<被试>.png ×3（每个被试一张）
用法（仓库根目录）: python preprocess/zhang2015-specific-face-model/viz/eve/side_view_v2_v3.py
"""
import json
import os
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

_PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_PROJECT))
sys.path.insert(0, str(_PROJECT / 'preprocess/zhang2015-specific-face-model/get_face_model'))

import face_model_core as core
from utils.logger import get_logger
from utils.normalization import vector_to_angles

log = get_logger('preprocess.specific_face_model.eve.side_view')

V2 = Path('/media/yanglinxuan/sfm/eve_specific_224')
V3 = Path('/media/yanglinxuan/sfm/eve_noroll_224')
RAW = Path('/media/yanglinxuan/zyx/EVE_dataset/eve_dataset')
SIDE_CAMS = ('webcam_l', 'webcam_r')
TAG_SHOW = {'v2': 'zhang', 'v3': 'ours'}   # 图上字样（缓存键仍 v2/v3）
N_SUBJ = 3
CACHE = Path(__file__).resolve().parent / 'cache_side_view.npz'
OUT = str(Path(__file__).resolve().parent / 'side_view_zhang_ours_{}.png')  # 按被试名
FONT = cv2.FONT_HERSHEY_SIMPLEX
FS = 0.52


def unit_from_angles(theta, phi):
    return np.array([-np.cos(theta) * np.sin(phi), -np.sin(theta),
                     -np.cos(theta) * np.cos(phi)])


def euler_ab(hR):
    """欧拉 (α,β) 度（extrinsic xyz，与 v3 协议一致）"""
    e = Rotation.from_matrix(hR).as_euler('xyz', degrees=True)
    return float(e[0]), float(e[1])


def _read_frame(subj, step, cam_name, frame):
    mp4 = RAW / subj / step / f'{cam_name}.mp4'
    cap = cv2.VideoCapture(str(mp4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()
    return img if ok else None


def collect():
    """3 个被试，每人同一同步组的 webcam_l+webcam_r 各两行 → 行记录列表"""
    rng = np.random.default_rng(0)
    rows_out = []
    for p3f in rng.permutation(sorted((V3 / 'train').glob('*.h5'))):
        if len(rows_out) >= 2 * N_SUBJ:
            break
        subj = p3f.stem
        p2f = V2 / 'train' / f'{subj}.h5'
        if not p2f.is_file():
            continue
        with h5py.File(p3f, 'r') as f3, h5py.File(p2f, 'r') as f2:
            cameras = json.loads(f3.attrs['cameras'])
            steps = json.loads(f3.attrs['steps'])
            model = np.array(f3.attrs['face_model'])
            fr3, ci3, st3 = (f3['frame_index'][:].ravel(),
                             f3['cam_index'][:].ravel(),
                             f3['step_index'][:].ravel())
            fr2, ci2, st2 = (f2['frame_index'][:].ravel(),
                             f2['cam_index'][:].ravel(),
                             f2['step_index'][:].ravel())
            # 同步组（basler c0 帧号 ÷2；webcam 帧号即同步帧）
            by_sync = {}
            for r in range(len(fr3)):
                c = int(ci3[r])
                by_sync.setdefault((int(fr3[r]) // 2 if c == 0 else int(fr3[r]),
                                    int(st3[r])), {})[cameras[c]] = r
            groups = [g for g in by_sync.values()
                      if all(k in g for k in SIDE_CAMS)]
            if not groups:
                continue
            g = groups[int(rng.integers(len(groups)))]
            for cam_name in SIDE_CAMS:
                r3 = g[cam_name]
                c, raw_f, si = int(ci3[r3]), int(fr3[r3]), int(st3[r3])
                step = steps[si]
                cand = [i for i in range(len(fr2))
                        if int(fr2[i]) == raw_f and int(ci2[i]) == c
                        and int(st2[i]) == si]
                if len(cand) != 1:
                    continue
                r2 = cand[0]
                lm3d = f3['face_landmarks_3d'][r3]
                matn2, gaze2 = f2['face_mat_norm'][r2], f2['face_gaze'][r2]
                matn3, gaze3 = f3['face_mat_norm'][r3], f3['face_gaze'][r3]
                hcs2 = np.degrees(f2['face_gaze_hcs'][r2])
                img = _read_frame(subj, step, cam_name, raw_f)
                if img is None:
                    continue
                hR, _ = core.kabsch(model, lm3d)
                gp_dir = matn2.T @ unit_from_angles(gaze2[0], gaze2[1])
                gpre = np.degrees(vector_to_angles(gp_dir))
                rows_out.append({
                    'cam': cam_name,
                    'id': f'{subj}/f{raw_f:04d}',
                    'raw': img,
                    'v2_patch': f2['face_patch'][r2],
                    'v3_patch': f3['face_patch'][r3],
                    'v2': {'head_post': euler_ab(matn2 @ hR),
                           'gaze_post': tuple(np.degrees(gaze2))},
                    'v3': {'head_post': euler_ab(matn3 @ hR),
                           'gaze_post': tuple(np.degrees(gaze3))},
                    'pre': {'head': euler_ab(hR), 'gaze': tuple(gpre)},
                    'hcs': tuple(hcs2),
                })
            log.info(f'采样 {subj}（{rows_out[-2]["id"] if len(rows_out) >= 2 else ""}'
                     f' l/r 同步组）' if len(rows_out) >= 2 else f'采样 {subj}')
    if len(rows_out) < 2 * N_SUBJ:
        raise SystemExit(f'仅采到 {len(rows_out)}/{2 * N_SUBJ} 行')
    return rows_out


def _fmt(v):
    return f'({v[0]:+6.1f},{v[1]:+6.1f})'


def render(rows):
    """两个子图（webcam_l / webcam_r，同一人同一时刻）上下叠放。

    每个子图：原始图像在左侧（高度 = 两个预处理图之和，长宽比不变），
    预处理图像在右侧分两行（上=zhang、下=ours），label 在最右侧与两行对齐。
    """
    PATCH, GAP, INNER, SUB_GAP = 224, 14, 8, 28
    BLOCK_H = 2 * PATCH + INNER
    HEAD_H = 30

    def tw(txt):
        return cv2.getTextSize(txt, FONT, FS, 1)[0][0]

    # 各子图：原图按 BLOCK_H 等高、保持长宽比缩放；label 宽度按实测文本
    raws, orig_ws, label_ws = [], [], []
    for r in rows:
        h0, w0 = r['raw'].shape[:2]
        ow = round(w0 * BLOCK_H / h0)
        raws.append(cv2.resize(r['raw'], (ow, BLOCK_H),
                               interpolation=cv2.INTER_AREA))
        orig_ws.append(ow)
        lw = 0
        for ver in ('v2', 'v3'):
            lw = max(lw,
                     tw(f"{TAG_SHOW[ver]}  {r['cam']}  {r['id']}"),
                     tw(f"head  pre{_fmt(r['pre']['head'])}"
                        f"  post{_fmt(r[ver]['head_post'])}"),
                     tw(f"gaze  pre{_fmt(r['pre']['gaze'])}"
                        f"  post{_fmt(r[ver]['gaze_post'])}"),
                     tw(f"HCS   {_fmt(r['hcs'])}"))
        label_ws.append(lw + 24)
    sub_w = [orig_ws[i] + GAP + PATCH + GAP + label_ws[i] for i in range(2)]
    W = GAP + max(sub_w) + GAP
    H = HEAD_H + 2 * BLOCK_H + SUB_GAP
    canvas = np.full((H, W, 3), 255, np.uint8)
    cv2.putText(canvas, f'EVE side view, same instant: {rows[0]["id"]}  '
                '(per subplot: raw | zhang / ours | labels)',
                (12, 21), FONT, 0.55, (60, 60, 60), 1, cv2.LINE_AA)

    def put(x, y, txt, c):
        cv2.putText(canvas, txt, (x, y), FONT, FS, c, 1, cv2.LINE_AA)

    def tag(x, y, txt):
        w = tw(txt) + 8
        cv2.rectangle(canvas, (x, y), (x + w, y + 20), (255, 255, 255), -1)
        put(x + 4, y + 15, txt, (30, 30, 30))

    y_sub = HEAD_H
    for s, r in enumerate(rows):
        if s > 0:                                   # 子图分隔线（横）
            cv2.line(canvas, (0, y_sub - SUB_GAP // 2),
                     (W, y_sub - SUB_GAP // 2), (200, 200, 200), 2)
        x_pat = GAP + orig_ws[s] + GAP
        x_lab = x_pat + PATCH + GAP
        # 左：原图（高度对应两个预处理图，长宽比不变）
        canvas[y_sub:y_sub + BLOCK_H, GAP:GAP + orig_ws[s]] = raws[s]
        tag(GAP, y_sub, r['cam'])
        # 右：预处理分两行
        for k, ver in enumerate(('v2', 'v3')):
            yy = y_sub + k * (PATCH + INNER)
            canvas[yy:yy + PATCH, x_pat:x_pat + PATCH] = r[f'{ver}_patch']
            tag(x_pat, yy, TAG_SHOW[ver])
            put(x_lab, yy + 22, f"{TAG_SHOW[ver]}  {r['cam']}  {r['id']}",
                (110, 110, 110))
            put(x_lab, yy + 54,
                f"head  pre{_fmt(r['pre']['head'])}"
                f"  post{_fmt(r[ver]['head_post'])}", (30, 30, 30))
            put(x_lab, yy + 86,
                f"gaze  pre{_fmt(r['pre']['gaze'])}"
                f"  post{_fmt(r[ver]['gaze_post'])}", (0, 0, 200))
            put(x_lab, yy + 118, f"HCS   {_fmt(r['hcs'])}", (0, 140, 0))
        y_sub += BLOCK_H + SUB_GAP

    subj = rows[0]['id'].split('/')[0]
    out = OUT.format(subj)
    cv2.imwrite(out, canvas)
    log.info(f'输出 {out}')


def main():
    if CACHE.is_file() and os.environ.get('CACHE_REFRESH', '') != '1':
        z = np.load(CACHE, allow_pickle=False)
        rows = json.loads(str(z['meta']))
        for i, r in enumerate(rows):           # 图像从缓存回填
            r['raw'] = z[f'raw{i}']
            r['v2_patch'] = z[f'v2_{i}']
            r['v3_patch'] = z[f'v3_{i}']
        log.info(f'缓存命中 {CACHE}（{len(rows)} 行）')
    else:
        rows = collect()
        kw = {'meta': np.array(json.dumps(
            [{k: v for k, v in r.items()
              if k not in ('raw', 'v2_patch', 'v3_patch')} for r in rows]))}
        for i, r in enumerate(rows):
            kw[f'raw{i}'], kw[f'v2_{i}'], kw[f'v3_{i}'] = \
                r['raw'], r['v2_patch'], r['v3_patch']
        np.savez_compressed(CACHE, **kw)
        log.info(f'缓存写入 {CACHE}（{len(rows)} 行）')

    for b in range(len(rows) // 2):           # 每个被试一张图（l/r 两行）
        render(rows[2 * b:2 * b + 2])


if __name__ == '__main__':
    main()
