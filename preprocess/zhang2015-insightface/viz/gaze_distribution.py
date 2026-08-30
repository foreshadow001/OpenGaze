"""insightface_224 四数据集 gaze 分布图（2026-08-30 重生成）

从 v1 预处理产物（`face_gaze` 字段，(pitch, yaw) 弧度）全量读取：
每数据集一张 pitch×yaw 2D 直方图热力图（含统计量标注），底部两条
pitch/yaw 分布曲线对比。同时刷新 gaze_distribution_stats.csv。

输出: 本目录 gaze_distribution_insightface_224.png / gaze_distribution_stats.csv
用法（仓库根目录）: python preprocess/zhang2015-insightface/viz/gaze_distribution.py
"""
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

_PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJECT))

from utils.logger import get_logger

log = get_logger('preprocess.insightface.viz.gaze_dist')

YLX = Path('/media/yanglinxuan/ylx')
OUT = Path(__file__).resolve().parent
DATASETS = [
    ('XGaze',        sorted(YLX.joinpath('xgaze_insightface_224').glob('subject*.h5'))),
    ('GazeCapture',  sorted(YLX.joinpath('gazecapture_insightface_224').glob('*/*.h5'))),
    ('MPIIFaceGaze', sorted(YLX.joinpath('mpiifacegaze_insightface_224').glob('p*.h5'))),
    ('EVE',          sorted(YLX.joinpath('eve_insightface_224').glob('*/*.h5'))),
]
LIM, BINS = 120, 240         # ±120°，热力图 1° 一格（曲线 2° 一格）
COLORS = {'XGaze': 'tab:blue', 'GazeCapture': 'tab:red',
          'EVE': 'tab:green', 'MPIIFaceGaze': '#DAA520'}   # 黄用 goldenrod 保可见性


def main():
    data, rows = {}, []
    grand = sum(len(fs) for _, fs in DATASETS)
    master = tqdm(total=grand, desc='总进度', unit='file', position=0, dynamic_ncols=True)
    for name, files in DATASETS:
        g, total = [], 0
        bar = tqdm(files, desc=f'读取 {name}', unit='file', position=1, dynamic_ncols=True,
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} '
                              '[{elapsed}<{remaining}, {rate_fmt}{postfix}]')
        for p in bar:
            with h5py.File(p, 'r') as f:
                a = np.asarray(f['face_gaze'], dtype=np.float64)
            g.append(a)
            total += len(a)
            bar.set_postfix({'累计样本': f'{total:,}'})
            master.update(1)
        bar.close()
        g = np.degrees(np.concatenate(g))
        data[name] = g
        stat = lambda v: (v.mean(), np.median(v), v.std(),
                          *np.percentile(v, [5, 95]))
        rows.append((name, len(g), *stat(g[:, 0]), *stat(g[:, 1])))
        log.info(f'{name}: {len(files)} 文件, N={len(g)}, '
                 f'pitch p50={rows[-1][3]:+.2f}° yaw p50={rows[-1][9]:+.2f}°')
    master.close()

    # ---- 布局：两条全宽分布曲线在上（纵向排布），2×2 热力图在下 ----
    fig = plt.figure(figsize=(14, 18))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.85, 0.85, 1, 1])
    curve_axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])]
    heat_axes = [fig.add_subplot(gs[i, j])
                 for i in (2, 3) for j in (0, 1)]

    # ---- 分布曲线（±120°，全宽） ----
    edges = np.linspace(-LIM, LIM, 121)          # 2° 一格
    centers = (edges[:-1] + edges[1:]) / 2
    for ax, k, lbl in zip(curve_axes, (0, 1), ('Pitch', 'Yaw')):
        for name, *_ in rows:
            w = np.ones(len(data[name])) / len(data[name])
            h, _ = np.histogram(np.clip(data[name][:, k], -LIM, LIM),
                                bins=edges, weights=w)
            h = gaussian_filter1d(h, sigma=2)               # 平滑（σ≈4°）
            ax.plot(centers, h * 100, label=name, lw=2.2,
                    color=COLORS.get(name))                 # % / degree
        ax.set_title(f'Gaze {lbl} Distribution (deg)', fontsize=12)
        ax.set_xlabel(f'{lbl} (deg)')
        ax.set_ylabel('% per degree')
        ax.set_xlim(-LIM, LIM)
        ax.set_xticks(np.arange(-120, 121, 30))             # 每 30° 一标
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    # ---- 热力图 2×2 ----
    for ax, (name, n, pm, pmd, ps, p5, p95, ym, ymd, ys, y5, y95) in \
            zip(heat_axes, rows):
        g = data[name]
        H, xe, ye = np.histogram2d(g[:, 0], g[:, 1], bins=BINS,
                                   range=[[-LIM, LIM], [-LIM, LIM]])
        im = ax.imshow(H.T, origin='lower', cmap='YlOrRd',
                       extent=[-LIM, LIM, -LIM, LIM], aspect='auto')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label='count per bin')
        ax.set_title(f'{name}  ({n:,} samples)', fontsize=12)
        ax.set_xlabel('Pitch (deg)')
        ax.set_ylabel('Yaw (deg)')
        ax.text(0.02, 0.98, '\n'.join([
            f'N = {n:,}',
            f'pitch: mean {pm:+.2f}  median {pmd:+.2f}  std {ps:.2f}',
            f'       p5 {p5:+.2f}  p95 {p95:+.2f}',
            f'yaw:   mean {ym:+.2f}  median {ymd:+.2f}  std {ys:.2f}',
            f'       p5 {y5:+.2f}  p95 {y95:+.2f}']),
            transform=ax.transAxes, va='top', fontsize=8, family='monospace',
            bbox=dict(fc='white', alpha=0.8, ec='0.7', pad=2))

    fig.suptitle('Gaze Distribution Comparison — insightface_224 '
                 '(Zhang2015 normalized, face_gaze 全量)', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png = OUT / 'gaze_distribution_insightface_224.png'
    fig.savefig(png, dpi=140)
    log.info(f'输出 {png}')

    # ---- CSV ----
    csv = OUT / 'gaze_distribution_stats.csv'
    with open(csv, 'w') as f:
        f.write('dataset,n,pitch_mean,pitch_median,pitch_std,pitch_p5,pitch_p95,'
                'yaw_mean,yaw_median,yaw_std,yaw_p5,yaw_p95\n')
        for r in rows:
            f.write(f'{r[0]},{r[1]},' + ','.join(f'{v:.4f}' for v in r[2:]) + '\n')
    log.info(f'输出 {csv}')


if __name__ == '__main__':
    main()
