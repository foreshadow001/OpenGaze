"""PoG 跨相机离散门控阈值统计（全量组，2026-08-30 留档）

对 EVE 全部同步组（44 被试，不抽样）计算 PoG 跨相机离散（px），统计各门控
阈值（5/10/15/20px）相对**整个数据集**的保留/丢失比例，供阈值决策。
离散定义与 pos_hcs_consistency.py 组门控一致：四相机 PoG 屏幕坐标到均值的
最大欧氏距离（px）。

输出（本目录）:
  pog_spread_gate_stats.csv   阈值 | 保留组 | 占全部组 | 占四相机齐 | 额外再弃
  pog_spread_gate_stats.png   全量离散分布 + 各阈值保留率
用法（仓库根目录）:
  /ssd/conda/envs/yanglinxuan/opengaze/bin/python \
  preprocess/zhang2015-specific-face-model/get_face_model/eve/metrics/frame_consistency/pog_spread_gate_stats.py
"""
import json
import sys
from pathlib import Path

import h5py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger

log = get_logger('preprocess.specific_face_model.eve.pog_spread_gate_stats')

LM = Path('/media/yanglinxuan/zyx/EVE_dataset/eve_dataset/landmarks')
RAW = Path('/media/yanglinxuan/zyx/EVE_dataset/eve_dataset')
CAMS = ['basler', 'webcam_l', 'webcam_c', 'webcam_r']
THRESHOLDS = (5, 10, 15, 20)


def collect():
    """全部同步组 → (相机不齐数, 四相机齐组的离散数组)"""
    n_total = n_incomplete = 0
    spreads = []
    for sp in ('train', 'test'):
        for p in tqdm(sorted(LM.joinpath(sp).glob('*.h5')),
                      desc=f'{sp} subjects', unit='subj'):
            subj = p.stem
            with h5py.File(p, 'r') as f:
                fr = f['frame_index'][:].ravel()
                ci = f['cam_index'][:].ravel()
                st = f['step_index'][:].ravel()
                steps = json.loads(f.attrs['steps'])
            by = {}
            for r in range(len(fr)):
                c, rf, si = int(ci[r]), int(fr[r]), int(st[r])
                by.setdefault((rf // 2 if c == 0 else rf, si), []).append((c, rf))
            pog_cache = {}
            for (sync_f, si), rows in by.items():
                n_total += 1
                if set(c for c, _ in rows) != set(range(4)):
                    n_incomplete += 1
                    continue
                step = steps[si]
                vals = []
                for c, rf in rows:
                    if (si, c) not in pog_cache:
                        with h5py.File(RAW / subj / step / f'{CAMS[c]}.h5',
                                       'r') as hf:
                            pog_cache[(si, c)] = np.array(
                                hf['face_PoG_tobii/data'])
                    dat = pog_cache[(si, c)]
                    vals.append(dat[min(rf, len(dat) - 1)])
                P = np.stack(vals)
                spreads.append(float(
                    np.max(np.linalg.norm(P - P.mean(0), axis=1))))
    return n_total, n_incomplete, np.array(spreads)


def main():
    n_total, n_incomplete, S = collect()
    n_complete = len(S)
    kept = {t: int((S <= t).sum()) for t in THRESHOLDS}
    kept20 = kept[20]

    csv = HERE / 'pog_spread_gate_stats.csv'
    with open(csv, 'w') as f:
        f.write('gate_px,kept_groups,pct_of_all,pct_of_complete,'
                'extra_drop_vs_20px\n')
        for t in THRESHOLDS:
            extra = '0.0%' if t == 20 else f'{(kept20 - kept[t]) / n_total:.1%}'
            f.write(f'{t},{kept[t]},{kept[t] / n_total:.3f},'
                    f'{kept[t] / n_complete:.3f},{extra}\n')
    log.info(f'全部同步组 {n_total:,} | 相机不齐 {n_incomplete:,} '
             f'({n_incomplete / n_total:.1%}) | 四相机齐 {n_complete:,}')
    for t in THRESHOLDS:
        extra = 0 if t == 20 else (kept20 - kept[t]) / n_total
        log.info(f'门控 ≤{t:>2}px: 保留 {kept[t]:>9,} '
                 f'({kept[t] / n_total:.1%} of 全部, '
                 f'{kept[t] / n_complete:.1%} of 四相机齐) | '
                 f'较 20px 再弃 {extra:.1%}')

    # 分布图（对数横轴，风格同 hcs_true6_dist / pog_spread_dist）
    fig, ax = plt.subplots(figsize=(9, 4.5))
    hi = 120.0
    lo = max(1e-2, S[S > 0].min() * 0.6)
    bins = np.logspace(np.log10(lo), np.log10(hi), 140)
    cnt, edges = np.histogram(S, bins=bins)
    dens = cnt / (len(S) * np.diff(np.log10(bins)))
    centers = np.sqrt(edges[:-1] * edges[1:])
    ax.plot(centers, dens, lw=1.8, color='#3b7dd8')
    ax.fill_between(centers, dens, alpha=0.15, color='#3b7dd8')
    ax.set_xscale('log')
    top = ax.get_ylim()[1]
    ys = [top * 0.97, top * 0.84, top * 0.71, top * 0.58]   # 每 4 个一错开
    marks = []
    for t in THRESHOLDS:
        marks.append((f'gate {t}px → keep {kept[t] / n_total:.1%}', t,
                      {5: 'tab:red', 10: 'tab:purple',
                       15: 'tab:brown', 20: 'gray'}[t]))
    marks.sort(key=lambda m: m[1])
    for i, (lbl, v, c) in enumerate(marks):
        ax.axvline(v, color=c, ls='-.', lw=1.6)
        ax.text(v * 1.05, ys[i % 4], lbl, color=c, fontsize=9,
                ha='left', va='top')
    ax.text(0.02, 0.97,
            f'all groups {n_total:,}\nincomplete cams {n_incomplete:,} '
            f'({n_incomplete / n_total:.1%})\ncomplete {n_complete:,}',
            transform=ax.transAxes, ha='left', va='top', fontsize=9,
            color='darkred', family='monospace',
            bbox=dict(fc='white', ec='darkred', alpha=0.85, pad=3))
    ax.set_xlim(lo, hi)
    ticks = [v for v in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100)
             if lo <= v <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{v:g}' for v in ticks])
    ax.minorticks_off()
    ax.set_xlabel('PoG cross-camera spread (px, log scale)')
    ax.set_ylabel('density')
    ax.set_title(f'EVE PoG spread distribution over ALL sync groups '
                 f'({len(S):,} complete, {n_total:,} total) & gate keep-rates')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png = HERE / 'pog_spread_gate_stats.png'
    fig.savefig(png, dpi=250)
    log.info(f'输出 {csv.name} / {png.name}')


if __name__ == '__main__':
    main()
