"""预处理公共：失败样本记录、一次运行的日志目录管理

一次预处理（preprocess.py 的一次调用）在 preprocess/log/ 下建一个运行目录：
    preprocess/log/<dataset>_<时间戳>/
    ├── run.log        分级日志（时间+级别+文件:行号，与训练日志同规范）
    └── failures.json  失败/跳过样本清单（原因分类汇总 + 明细）
"""
import json
import os
import time
from collections import Counter

from utils.logger import attach_file_handler, get_logger

LOG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log')


class FailureRecorder:
    """记录处理失败/跳过的样本。

    原脚本直接 continue 跳过失败帧（读图失败/检不到脸/无标注）不留痕，
    导致 h5 样本数与标注数对不上无从核对；此类在此逐条记录，
    结束后保存 failures.json 供核对数据完整性。
    """

    def __init__(self):
        self.records = []

    def add(self, subject, sample, reason):
        """subject: 被试标识；sample: 帧标识（路径/帧号）；reason: 失败原因（简短英文 key）"""
        self.records.append({'subject': str(subject), 'sample': str(sample),
                             'reason': reason})

    def save(self, path):
        by_reason = Counter(r['reason'] for r in self.records)
        doc = {
            'total_failed': len(self.records),
            'by_reason': dict(by_reason),
            'samples': self.records,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        return doc


def new_run_dir(dataset_name):
    """创建本次预处理的运行目录并挂载文件日志，返回目录路径"""
    run_dir = os.path.join(
        LOG_ROOT, f'{dataset_name}_{time.strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(run_dir, exist_ok=True)
    attach_file_handler(os.path.join(run_dir, 'run.log'), append=False)
    get_logger('preprocessor').info(f'预处理运行目录: {run_dir}')
    return run_dir
