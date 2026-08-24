"""预处理入口

用法：
    python preprocess.py --dataset mpiifacegaze --method zhang2015-insightface
    python preprocess.py --dataset mpiifacegaze --method zhang2015-insightface --set 'subjects=["p00"]'
    python preprocess.py --dataset xgaze --method zhang2015-insightface --set 'subjects=[0, 3]'

预处理器按管线组织在 preprocess/<method>/ 下（如 zhang2015-insightface，
目录名含连字符不是合法包名，故由本入口用 importlib 按文件路径加载，
脚本命名约定 normalize_<dataset>.py）。

每次运行在 preprocess/log/<dataset>_<时间戳>/ 下留档：
    run.log        分级日志（与训练日志同规范）
    failures.json  失败/跳过样本清单（原因汇总 + 明细）
"""
import argparse
import importlib.util
import os
import sys

from preprocess.common import FailureRecorder, new_run_dir
from utils.config import apply_overrides, load_yaml, yaml_to_ns
from utils.logger import get_logger

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PREPROCESS_CONFIG_DIR = os.path.join(PROJECT_ROOT, 'configs', 'preprocess')
PREPROCESS_ROOT = os.path.join(PROJECT_ROOT, 'preprocess')

log = get_logger('preprocess')


def _load_module(name, path):
    """按文件路径加载预处理模块（其所在目录名含连字符，不可 import）"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description='OpenGaze 数据预处理')
    parser.add_argument('--dataset', required=True,
                        help='数据集名，对应 configs/preprocess/<name>.yaml，'
                             '如 xgaze / mpiifacegaze')
    parser.add_argument('--method', required=True,
                        help='预处理管线名，对应 preprocess/<method>/ 目录，'
                             '如 zhang2015-insightface')
    parser.add_argument('--set', nargs='+', default=[], metavar='KEY=VALUE',
                        help='覆盖配置项，点路径，如 --set subjects=\'["p00"]\'')
    args = parser.parse_args()

    config_path = os.path.join(PREPROCESS_CONFIG_DIR, f'{args.dataset}.yaml')
    if not os.path.exists(config_path):
        raise SystemExit(f'数据集配置不存在: {config_path}')
    config = yaml_to_ns(load_yaml(config_path))
    apply_overrides(config, args.set)

    # 管线目录下按数据集命名约定定位脚本
    script = os.path.join(PREPROCESS_ROOT, args.method,
                          f'normalize_{args.dataset}.py')
    if not os.path.exists(script):
        raise SystemExit(f'管线 {args.method} 下无 {args.dataset} 预处理器: {script}')

    run_dir = new_run_dir(args.dataset)
    log.info(f'数据集: {args.dataset} | 管线: {args.method} | 配置: {config_path}')
    for key, value in vars(config).items():
        log.info(f'  {key}: {value}')

    module = _load_module(f'normalize_{args.dataset}', script)
    recorder = FailureRecorder()
    module.run(config, recorder)

    failure_path = os.path.join(run_dir, 'failures.json')
    doc = recorder.save(failure_path)
    log.info(f"失败/跳过样本: {doc['total_failed']} {doc['by_reason']}")
    log.info(f'日志与失败清单已保存: {run_dir}')


if __name__ == '__main__':
    main()
