"""分级日志与实验目录管理

日志规范（见 CLAUDE.md）：所有输出必须经 get_logger(__name__)，禁止 print / tqdm.write。
    - 可溯源：格式含 时间 + 级别 + 文件:行号
    - 分级别：DEBUG / INFO / WARNING / ERROR（根 logger 默认 INFO）
    - 双写：console + exp/expNN/logs/run.log（由 ExperimentLogger 挂载文件 handler，
      该实验进程内所有模块的日志都会写入，测试/续训以 append 模式续写同一文件）

实验目录 exp/expNN 自包含：config.yaml / method.yaml / ckpt/ / logs/（见 STRUCTURE.md 3.3）。
"""
import logging
import os
import re
import shutil
import time

import yaml
from torch.utils.tensorboard import SummaryWriter

ROOT_LOGGER_NAME = 'opengaze'

# console 短格式（时分秒），文件长格式（含日期，长期留档）
_CONSOLE_FMT = '%(asctime)s | %(levelname).1s | %(filename)s:%(lineno)d | %(message)s'
_FILE_FMT = '%(asctime)s | %(levelname)-7s | %(filename)s:%(lineno)d | %(message)s'


def get_logger(name=None):
    """获取模块 logger：get_logger(__name__) → opengaze.trainers.trainer 等"""
    if not name or name == ROOT_LOGGER_NAME:
        return logging.getLogger(ROOT_LOGGER_NAME)
    if name.startswith(ROOT_LOGGER_NAME + '.'):
        return logging.getLogger(name)
    return logging.getLogger(f'{ROOT_LOGGER_NAME}.{name}')


def _ensure_root_logger():
    """确保根 logger 存在且带 console handler（默认 INFO 级）"""
    root = logging.getLogger(ROOT_LOGGER_NAME)
    if root.level == logging.NOTSET:
        root.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_CONSOLE_FMT, datefmt='%H:%M:%S'))
        root.addHandler(handler)
    return root


def attach_file_handler(log_path, append=True):
    """把文件 handler 挂到根 logger：此后所有模块日志追加写入 log_path"""
    root = _ensure_root_logger()
    handler = logging.FileHandler(log_path, mode='a' if append else 'w',
                                  encoding='utf-8')
    handler.setFormatter(logging.Formatter(_FILE_FMT, datefmt='%Y-%m-%d %H:%M:%S'))
    root.addHandler(handler)
    return handler


def next_exp_id(exp_root):
    """扫描 exp/ 下已有 expNN，返回下一个编号名（exp00 起步）"""
    os.makedirs(exp_root, exist_ok=True)
    ids = []
    for name in os.listdir(exp_root):
        m = re.fullmatch(r'exp(\d+)', name)
        if m and os.path.isdir(os.path.join(exp_root, name)):
            ids.append(int(m.group(1)))
    return f'exp{(max(ids) + 1) if ids else 0:02d}'


class ExperimentLogger:
    """实验（或实验内子运行）的输出管理

    目录结构：
        exp/expNN/                      一次实验（如 MPII 的完整 LOO 评测）
        ├── config.yaml / method.yaml   配置快照，实验首次创建时写
        ├── <run>/                      子运行（LOO 的 fold_00~fold_14、all 等）
        │   ├── ckpt/                   该子运行的 checkpoint
        │   ├── logs/                   tensorboard + run.log
        │   └── test_result_*.json
        └── ckpt/ logs/ …               无子运行时直接放 exp 目录（如 xgaze）

    Args:
        exp_id: None 时自动递增分配（仅 mode='create'）
        run_name: 子运行名；None = 直接使用 exp 目录
        mode: 'create' 新建——exp 目录不存在则创建（created_exp=True，快照在此时写），
                      已存在且带 run_name 则复用（追加子运行）；run 目录已存在则拒绝
              'append' 打开已有 run 目录（断点续训；测试日志同模式追加）
    """

    def __init__(self, exp_root, exp_id=None, run_name=None, mode='create'):
        if exp_id is None:
            if mode != 'create':
                raise ValueError('append 模式必须显式指定 exp_id')
            exp_id = next_exp_id(exp_root)
        if not re.fullmatch(r'exp\d+', exp_id):
            raise ValueError(f'实验目录名应为 expNN 形式，收到: {exp_id}')
        if run_name is not None and not re.fullmatch(r'[A-Za-z0-9_]+', run_name):
            raise ValueError(f'子运行名仅允许字母数字下划线，收到: {run_name}')

        self.exp_id = exp_id
        self.run_name = run_name
        self.exp_dir = os.path.join(exp_root, exp_id)
        self.run_dir = os.path.join(self.exp_dir, run_name) if run_name else self.exp_dir
        self.ckpt_dir = os.path.join(self.run_dir, 'ckpt')
        self.log_dir = os.path.join(self.run_dir, 'logs')

        self.created_exp = False
        if mode == 'create':
            if run_name is None:
                # 无子运行：exp 目录即 run 目录，必须不存在
                if os.path.exists(self.exp_dir):
                    raise FileExistsError(
                        f'实验目录已存在，拒绝覆盖: {self.exp_dir}')
                os.makedirs(self.ckpt_dir)
                os.makedirs(self.log_dir)
                self.created_exp = True
            else:
                # 带子运行：exp 目录可复用（首次创建时写快照），run 目录必须不存在
                if not os.path.exists(self.exp_dir):
                    os.makedirs(self.exp_dir)
                    self.created_exp = True
                if os.path.exists(self.run_dir):
                    raise FileExistsError(
                        f'子运行目录已存在，拒绝覆盖: {self.run_dir}')
                os.makedirs(self.ckpt_dir)
                os.makedirs(self.log_dir)
        elif mode == 'append':
            if not os.path.isdir(self.run_dir):
                raise FileNotFoundError(f'实验/子运行目录不存在: {self.run_dir}')
        else:
            raise ValueError(f'未知 mode: {mode}')

        # tensorboard：续开时新建事件文件，曲线按恢复的 train_iter 续接
        self.writer = SummaryWriter(log_dir=self.log_dir)
        # 所有模块日志追加写入该 run 的 run.log
        self._file_handler = attach_file_handler(
            os.path.join(self.log_dir, 'run.log'), append=(mode == 'append'))

        where = self.exp_dir + (f'/{run_name}' if run_name else '')
        self.info(f'实验目录: {where}（{"续开" if mode == "append" else "新建"}）')

    def info(self, msg):
        _ensure_root_logger()
        logging.getLogger(ROOT_LOGGER_NAME).info(msg)

    def add_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def save_config(self, config, method_yaml_path=None):
        """落盘配置快照。

        快照两级：
            run 级  exp/expNN/[run/]config.yaml —— 本次运行 --set 后的实际配置
                    （resume / test 读取，保证子运行如 LOO 各折的 fold 划分正确）
            顶层    exp/expNN/config.yaml —— 仅实验首次创建时写（require_exp 匹配、
                    无子运行时的唯一快照）
        method.yaml 副本仅在实验首次创建时写入顶层。
        """
        from utils.config import ns_to_dict
        snapshot = ns_to_dict(config)
        snapshot['meta'] = {
            'exp_id': self.exp_id,
            'run': self.run_name,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        run_cfg_path = os.path.join(self.run_dir, 'config.yaml')
        with open(run_cfg_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(snapshot, f, allow_unicode=True, sort_keys=False)

        if self.created_exp:
            if self.run_dir != self.exp_dir:
                shutil.copyfile(run_cfg_path,
                                os.path.join(self.exp_dir, 'config.yaml'))
            if method_yaml_path is not None:
                shutil.copyfile(method_yaml_path,
                                os.path.join(self.exp_dir, 'method.yaml'))
        self.info(f'配置快照已保存: {run_cfg_path}')

    def close(self):
        self.writer.close()
        logging.getLogger(ROOT_LOGGER_NAME).removeHandler(self._file_handler)
        self._file_handler.close()


def find_latest_ckpt(exp_dir, run_name=None):
    """exp 目录（或其子运行目录）ckpt/ 下编号最大的 epoch_*_ckpt.pth"""
    ckpt_dir = os.path.join(exp_dir, run_name, 'ckpt') if run_name \
        else os.path.join(exp_dir, 'ckpt')
    best_path, best_epoch = None, -1
    for name in os.listdir(ckpt_dir):
        m = re.fullmatch(r'epoch_(\d+)_ckpt\.pth', name)
        if m and int(m.group(1)) > best_epoch:
            best_path, best_epoch = os.path.join(ckpt_dir, name), int(m.group(1))
    if best_path is None:
        raise FileNotFoundError(f'{ckpt_dir} 下没有 checkpoint')
    return best_path
