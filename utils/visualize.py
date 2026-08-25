"""实验结果可视化（静态 PNG，matplotlib Agg 后端）

数据源均为实验目录的自包含产物，无外部依赖：
    exp/expNN/fold_*/test_result_<dataset>.json   LOO 各折测试结果
    exp/expNN/fold_*/logs/                        tensorboard 标量（各折训练曲线）
    exp/expNN/all/logs/                           全量训练曲线
    exp/expNN/test_result_<dataset>.json          普通实验（无子运行）的测试结果

图规范（dataviz）：单系列柱状图单色 + 均值参考线；多折线以"组"为 identity
（LOO 各折 = 淡色同系，all = 强调色），≤2 图例；单轴；recessive 网格；
均值直接标注；数值标注用文本色而非系列色。
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# 调色（中性蓝主色 + 灰阶辅助；文本用近黑而非系列色）
C_BAR = '#4C78A8'        # LOO 折柱（单系列）
C_BAR_EDGE = '#3A5F8A'
C_MEAN = '#C1554F'       # 均值参考线（唯一第二色，语义=统计量）
C_FOLD_LINE = '#B7C4D6'  # 各折训练曲线（组：淡色）
C_ALL_LINE = '#4C78A8'   # all run 曲线（组：强调）
C_TEXT = '#333333'
C_GRID = '#E4E4E4'

plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': '#CCCCCC', 'axes.labelcolor': C_TEXT,
    'xtick.color': '#666666', 'ytick.color': '#666666',
    'axes.titlesize': 12, 'figure.facecolor': 'white',
})


def load_event_scalars(log_dir):
    """读 tensorboard 事件目录 → {tag: (steps, values)}；无事件返回空 dict"""
    from tensorboard.backend.event_processing.event_accumulator import \
        EventAccumulator
    if not os.path.isdir(log_dir) or not os.listdir(log_dir):
        return {}
    try:
        acc = EventAccumulator(log_dir)
        acc.Reload()
        return {tag: ([s.step for s in acc.Scalars(tag)],
                      [s.value for s in acc.Scalars(tag)])
                for tag in acc.Tags()['scalars']}
    except Exception:
        return {}


def collect_runs(exp_dir):
    """发现实验的全部运行：返回 [(run_name, run_dir)]，fold_* 排序在前，all 随后"""
    runs = []
    for name in sorted(os.listdir(exp_dir)):
        d = os.path.join(exp_dir, name)
        if os.path.isdir(d) and name.startswith('fold_'):
            runs.append((name, d))
    all_dir = os.path.join(exp_dir, 'all')
    if os.path.isdir(all_dir):
        runs.append(('all', all_dir))
    return runs


def collect_test_results(exp_dir):
    """收集全部 test_result_*.json → [(run_name_or_None, 结果 dict)]"""
    results = []
    for run_name, run_dir in collect_runs(exp_dir):
        for f in sorted(os.listdir(run_dir)):
            if f.startswith('test_result_') and f.endswith('.json'):
                results.append((run_name, json.load(open(os.path.join(run_dir, f)))))
    for f in sorted(os.listdir(exp_dir)):          # 普通实验：顶层结果
        if f.startswith('test_result_') and f.endswith('.json'):
            results.append((None, json.load(open(os.path.join(exp_dir, f)))))
    return results


def plot_loo_errors(test_results, out_path, dataset_name):
    """LOO 各折误差柱状图 + 均值参考线（test_results: [(run, dict)]）"""
    folds = sorted((r, d['gaze_error_deg']) for r, d in test_results
                   if r and r.startswith('fold_'))
    if not folds:
        return False
    names = [r.replace('fold_', 'f') for r, _ in folds]
    vals = [v for _, v in folds]
    mean = sum(vals) / len(vals)
    std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    bars = ax.bar(names, vals, width=0.62, color=C_BAR, edgecolor=C_BAR_EDGE,
                  linewidth=0.5, zorder=3)
    ax.axhline(mean, color=C_MEAN, linewidth=1.4, linestyle='--', zorder=4)
    ax.text(len(names) - 0.4, mean, f'mean {mean:.2f}°', color=C_MEAN,
            fontsize=10, va='bottom', ha='right', fontweight='bold')
    for bar, v in zip(bars, vals):                  # 数值标注（文本色，非系列色）
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.03, f'{v:.2f}',
                ha='center', va='bottom', fontsize=7.5, color=C_TEXT)
    ax.set_xlabel('leave-one-out fold (test subject p00-p14)')
    ax.set_ylabel('gaze error (deg)')
    ax.set_title(f'LOO per-fold gaze error — {dataset_name} '
                 f'(mean {mean:.2f}° ± {std:.2f}°, n={len(folds)})')
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(axis='y', color=C_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def _is_epoch_steps(steps):
    """判别事件 step 语义：新格式每 epoch 一个点（连续整数）；旧格式为累计迭代数"""
    return steps == list(range(steps[0], steps[0] + len(steps)))


def plot_training_curves(runs, out_path):
    """训练曲线：LOO 各折淡色组 + all 强调线（Loss/gaze；有 Error/train 时双联图）"""
    scalars = {name: load_event_scalars(os.path.join(d, 'logs'))
               for name, d in runs}
    tags = [t for t in ('Loss/gaze', 'Error/train')
            if any(t in s for s in scalars.values())]
    if not tags:
        return False
    fig, axes = plt.subplots(1, len(tags), figsize=(5.2 * len(tags), 4), dpi=150)
    if len(tags) == 1:
        axes = [axes]
    for ax, tag in zip(axes, tags):
        for name, s in scalars.items():
            if tag not in s:
                continue
            steps, vals = s[tag]
            if name == 'all':
                ax.plot(steps, vals, color=C_ALL_LINE, linewidth=1.8,
                        label='all (full training)', zorder=3)
            else:
                ax.plot(steps, vals, color=C_FOLD_LINE, linewidth=1.0,
                        zorder=2)
        ax.plot([], [], color=C_FOLD_LINE, linewidth=1.0, label='LOO folds')
        first_steps = next(s[tag][0] for s in scalars.values() if tag in s)
        ax.set_xlabel('epoch' if _is_epoch_steps(first_steps) else 'iteration')
        ax.set_ylabel(tag)
        ax.grid(color=C_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    fig.suptitle('training curves', fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return True


def write_summary(test_results, runs, out_path):
    """文本汇总：LOO mean±std、逐折明细、各运行说明"""
    lines = []
    folds = sorted((r, d) for r, d in test_results if r and r.startswith('fold_'))
    if folds:
        vals = [d['gaze_error_deg'] for _, d in folds]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        lines.append(f'LOO mean ± std: {mean:.2f}° ± {std:.2f}° '
                     f'({len(folds)} folds)')
        lines.append('-' * 56)
        for r, d in folds:
            lines.append(f"  {r}: {d['gaze_error_deg']:.2f}°  "
                         f"({d['num_samples']} samples)")
    for r, d in test_results:
        if r is None or r == 'all':
            lines.append(f"  [{r or 'experiment'}] test on {d['test_dataset']}: "
                         f"{d['gaze_error_deg']:.2f}° ({d['num_samples']} samples)")
    lines.append('-' * 56)
    for name, d in runs:
        lines.append(f'  run {name}: '
                     + ('tested' if any(r == name for r, _ in test_results)
                        else 'trained (no test result)'))
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return lines
