"""实验可视化入口：指定实验目录，生成图表到 expNN/viz/

用法：
    python viz.py --exp exp00              # LOO 实验（MPII）：折误差图 + 训练曲线 + 汇总
    python viz.py --exp exp01              # 普通实验（如 xgaze）：训练曲线 + 测试结果汇总

输出（有数据才生成）：
    expNN/viz/loo_errors.png      LOO 各折误差柱状图 + mean±std
    expNN/viz/training_curves.png 训练曲线（各折淡色组 + all 强调线）
    expNN/viz/summary.txt         文本汇总
"""
import argparse
import os

from utils.logger import get_logger
from utils.visualize import (collect_runs, collect_test_results,
                             plot_loo_errors, plot_training_curves,
                             write_summary)

EXP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exp')
log = get_logger('viz')


def main():
    parser = argparse.ArgumentParser(description='OpenGaze 实验可视化')
    parser.add_argument('--exp', required=True, help='实验目录名，如 exp00')
    args = parser.parse_args()

    exp_dir = os.path.join(EXP_ROOT, args.exp)
    if not os.path.isdir(exp_dir):
        raise SystemExit(f'实验目录不存在: {exp_dir}')
    out_dir = os.path.join(exp_dir, 'viz')
    os.makedirs(out_dir, exist_ok=True)
    log.info(f'实验: {args.exp} → 输出 {out_dir}')

    runs = collect_runs(exp_dir)
    test_results = collect_test_results(exp_dir)
    if not runs and not test_results:
        raise SystemExit('实验目录下无子运行与测试结果，无可视化内容')

    dataset = (test_results[0][1].get('test_dataset', 'unknown')
               if test_results else 'unknown')

    produced = []
    if plot_loo_errors(test_results, os.path.join(out_dir, 'loo_errors.png'),
                       dataset):
        produced.append('loo_errors.png')
    if plot_training_curves(runs, os.path.join(out_dir, 'training_curves.png')):
        produced.append('training_curves.png')
    lines = write_summary(test_results, runs, os.path.join(out_dir, 'summary.txt'))
    produced.append('summary.txt')

    log.info(f'已生成: {", ".join(produced)}')
    for line in lines[:8]:
        log.info(f'  {line}')


if __name__ == '__main__':
    main()
