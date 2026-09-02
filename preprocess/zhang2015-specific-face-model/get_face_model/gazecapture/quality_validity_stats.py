"""GC 官方有效性标注统计：四条件同时满足的帧占比（2026-08-30）

背景：insightface 在脸部遮挡时也可能检出人脸（影响图像质量）。统计官方
iTracker 检测的四条件同时满足帧占比：
  dot 有效（DotNum != -1 且 XCam/YCam 非 None）
  ∧ appleFace.IsValid ∧ appleLeftEye.IsValid ∧ appleRightEye.IsValid
（eye 有效 ⇒ face 有效已实证，四条件即官方最强质量门。）

缓存 validity_cache.npz（逐 session 计数）：命中跳过解析；
GC_QUALITY_REFRESH=1 强制重算。

输出（本目录）:
  validity_stats.csv    逐 session（各条件帧数 + 四条件帧数 + 占比）
  汇总打印：总体/分 train-test/分朝向的各条件与四条件占比
用法（仓库根目录）:
  /ssd/conda/envs/yanglinxuan/opengaze/bin/python \
  preprocess/zhang2015-specific-face-model/get_face_model/gazecapture/quality_validity_stats.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger

log = get_logger('preprocess.specific_face_model.gc.validity_stats')

RAW = Path('/media/yanglinxuan/zyx/GazeCapture')
CACHE = HERE / 'validity_cache.npz'
SESSIONS = sorted(p.name for p in RAW.iterdir()
                  if p.is_dir() and p.name.isdigit())
SPLITS = {}                       # session -> 'train'/'test'（info.json Dataset）


def scan_session(s):
    rec = RAW / s
    try:
        dot = json.load(open(rec / 'dotInfo.json'))
        face = json.load(open(rec / 'appleFace.json'))
        le = json.load(open(rec / 'appleLeftEye.json'))
        re_ = json.load(open(rec / 'appleRightEye.json'))
        scr = json.load(open(rec / 'screen.json'))
        info = json.load(open(rec / 'info.json'))
    except Exception:
        return None
    n = min(len(dot['DotNum']), len(face['IsValid']),
            len(le['IsValid']), len(re_['IsValid']),
            len(scr['Orientation']))
    dot_v = np.array([(d != -1 and x is not None and y is not None)
                      for d, x, y in zip(dot['DotNum'][:n],
                                          dot['XCam'][:n],
                                          dot['YCam'][:n])])
    f_v = np.array(face['IsValid'][:n], dtype=bool)
    l_v = np.array(le['IsValid'][:n], dtype=bool)
    r_v = np.array(re_['IsValid'][:n], dtype=bool)
    ori = np.array(scr['Orientation'][:n])
    return {'n': n, 'dot': dot_v, 'face': f_v, 'leye': l_v,
            'reye': r_v, 'all4': dot_v & f_v & l_v & r_v, 'ori': ori,
            'split': info.get('Dataset', '')}


def main():
    if CACHE.is_file() and not os.environ.get('GC_QUALITY_REFRESH'):
        z = np.load(CACHE, allow_pickle=False)
        log.info(f'缓存命中 {CACHE.name}')
    else:
        rows = {}
        for s in tqdm(SESSIONS, desc='sessions', unit='sess'):
            r = scan_session(s)
            if r is not None:
                rows[s] = r
        np.savez_compressed(
            CACHE,
            sessions=np.array(sorted(rows)),
            **{f'{s}__{k}': v for s, r in rows.items()
               for k, v in r.items() if k != 'split'},
            splits=np.array([rows[s]['split'] for s in sorted(rows)]))
        log.info(f'缓存写入 {CACHE.name}（{len(rows)} session）')
        z = np.load(CACHE, allow_pickle=False)

    sessions = list(z['sessions'])
    splits = z['splits']
    csv = HERE / 'validity_stats.csv'
    with open(csv, 'w') as f:
        f.write('session,split,n_frames,dot_valid,face_valid,left_eye_valid,'
                'right_eye_valid,all4,all4_pct\n')
        tot = {k: 0 for k in ('n', 'dot', 'face', 'leye', 'reye', 'all4')}
        by_split = {}
        by_ori = {}
        for s, sp in zip(sessions, splits):
            r = {k: z[f'{s}__{k}'] for k in ('n', 'dot', 'face', 'leye',
                                             'reye', 'all4', 'ori')}
            n = int(r['n'])
            c = {k: int(r[k].sum()) for k in ('dot', 'face', 'leye',
                                              'reye', 'all4')}
            f.write(f'{s},{sp},{n},{c["dot"]},{c["face"]},{c["leye"]},'
                    f'{c["reye"]},{c["all4"]},{c["all4"] / max(n, 1):.4f}\n')
            for k in tot:
                tot[k] += (n if k == 'n' else c[k])
            acc = by_split.setdefault(sp, {k: 0 for k in tot})
            for k in tot:
                acc[k] += (n if k == 'n' else c[k])
            for o in (1, 2, 3, 4):
                m = r['ori'] == o
                if m.any():
                    acc = by_ori.setdefault(o, {k: 0 for k in tot})
                    acc['n'] += int(m.sum())
                    for k in ('dot', 'face', 'leye', 'reye', 'all4'):
                        acc[k] += int(r[k][m].sum())

    def pct(a, b):
        return f'{a / max(b, 1) * 100:5.2f}%'

    lines = [f'sessions={len(sessions)}  总帧数={tot["n"]:,}',
             f'  dot 有效          {tot["dot"]:>9,}  {pct(tot["dot"], tot["n"])}',
             f'  appleFace 有效    {tot["face"]:>9,}  {pct(tot["face"], tot["n"])}',
             f'  appleLeftEye 有效 {tot["leye"]:>9,}  {pct(tot["leye"], tot["n"])}',
             f'  appleRightEye 有效{tot["reye"]:>9,}  {pct(tot["reye"], tot["n"])}',
             f'  四条件同时满足    {tot["all4"]:>9,}  {pct(tot["all4"], tot["n"])}',
             '', '按 split：']
    for sp in sorted(by_split):
        a = by_split[sp]
        lines.append(f'  {sp:5s} n={a["n"]:>9,}  四条件 {a["all4"]:>9,} '
                     f'{pct(a["all4"], a["n"])}')
    lines.append('')
    lines.append('按朝向：')
    for o in (1, 2, 3, 4):
        if o in by_ori:
            a = by_ori[o]
            lines.append(f'  ori{o} n={a["n"]:>9,}  四条件 {a["all4"]:>9,} '
                         f'{pct(a["all4"], a["n"])}')
    summary = '\n'.join(lines)
    log.info('\n' + summary)
    (HERE / 'validity_summary.txt').write_text(summary + '\n')
    log.info(f'输出 {csv.name} / validity_summary.txt')


if __name__ == '__main__':
    main()
