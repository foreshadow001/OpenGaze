"""OpenGaze 入口

训练（自动创建 exp/expNN，配置快照自动落盘）：
    python main.py --dataset zhang2015-insightface/xgaze --method resnet50

断点续训（只指定实验目录；配置以 exp01 快照为准，从最新 ckpt 完整恢复）：
    python main.py --resume exp01

测试（加载 exp00 中最新 epoch 的 checkpoint）：
    python main.py --dataset zhang2015-insightface/xgaze --method resnet50 --test --exp exp00

跨数据集评测（ckpt 取自 exp00，测试集按 --dataset 现场构建）：
    python main.py --dataset zhang2015-insightface/mpiifacegaze --method resnet50 --test --exp exp00

临时覆盖配置项：
    python main.py --dataset zhang2015-insightface/xgaze --method resnet50 --set method.train.epochs=2

多卡训练（DDP，脚本内已封装；测试始终单卡）：
    python -m torch.distributed.run --nproc_per_node=4 main.py --dataset zhang2015-insightface/xgaze --method resnet50
"""
import argparse
import os

import numpy as np
import torch
import torch.distributed as dist

from datasets import build_test_loader, build_train_loader
from trainers import Trainer
from utils.config import (PROJECT_ROOT, apply_overrides, load_config,
                          load_dataset_config, load_yaml, yaml_to_ns)
from utils.logger import ExperimentLogger, attach_file_handler, find_latest_ckpt, get_logger

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.join(PROJECT_ROOT, 'exp')

log = get_logger('main')


def parse_args():
    parser = argparse.ArgumentParser(description='OpenGaze: 通用视线估计训练/测试平台')
    parser.add_argument('--dataset', default=None,
                        help='configs/datasets/ 下的配置名（可含管线子路径，'
                             '如 zhang2015-insightface/xgaze）；'
                             '续训时可省略（用实验快照）')
    parser.add_argument('--method', default=None,
                        help='configs/methods/ 下的配置名，如 resnet50；'
                             '测试/续训时可省略，默认用实验快照中的 method')
    parser.add_argument('--test', action='store_true', help='测试模式')
    parser.add_argument('--label', choices=['ccs', 'hcs'], default=None,
                        help='训练/测试的视线标签：ccs=归一化相机系（face_gaze，'
                             '默认），hcs=头架系（face_gaze_hcs，v3 产物）。'
                             '训练时写入快照；测试时快照优先（跨数据集评测'
                             '自动沿用源实验标签），显式指定则覆盖')
    parser.add_argument('--resume', default=None, metavar='EXPNN',
                        help='断点续训：从指定实验目录的最新 ckpt 恢复，'
                             '配置以该实验的快照为准（如 --resume exp01）')
    parser.add_argument('--run', default=None, metavar='NAME',
                        help='子运行名（实验目录下的子目录，如 LOO 的 fold_00、all）。'
                             '训练时在 exp 目录下新建；测试/续训时定位子运行的 ckpt；'
                             '省略则直接使用 exp 目录本身')
    parser.add_argument('--exp', default=None,
                        help='新训练的目录名（expNN），不给则自动递增创建，'
                             '已存在则拒绝覆盖')
    parser.add_argument('--ckpt', default=None,
                        help='测试时指定 ckpt 文件名（位于 exp 目录 ckpt/ 下），'
                             '默认取最新 epoch')
    parser.add_argument('--set', nargs='+', default=[], metavar='KEY=VALUE',
                        help='覆盖配置项，点路径，如 --set method.train.epochs=2 '
                             'dataset.dataloader.num_workers=2')
    return parser.parse_args()


def _init_distributed():
    """torchrun 多卡训练时初始化分布式进程组（WORLD_SIZE>1）；单进程返回 (False, None)

    多卡启动方式（脚本内已封装，见 scripts/common.sh 的 py()）：
        python -m torch.distributed.run --nproc_per_node=4 main.py --dataset ... --method ...
    返回 (distributed, device)：device 为本进程专属卡 cuda:local_rank。
    """
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    if world_size <= 1:
        return False, None
    if not torch.cuda.is_available():
        raise SystemExit('torchrun 多卡训练需要 CUDA（未检测到可用 GPU）')
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    return True, torch.device(f'cuda:{local_rank}')


def _apply_common_env():
    """平台公共配置 configs/common.yaml → 环境变量。

    gpus → CUDA_VISIBLE_DEVICES：仅当环境未显式设置时（scripts/common.sh 的
    py() 启动器/手动 torchrun 已设则尊重之）。必须在任何 CUDA 初始化之前调用。
    """
    if os.environ.get('CUDA_VISIBLE_DEVICES'):
        return
    path = os.path.join(PROJECT_ROOT, 'configs', 'common.yaml')
    if not os.path.exists(path):
        return
    gpus = load_yaml(path).get('gpus')
    if gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpus)
        # 此时尚无任何 handler（ExperimentLogger/attach_file_handler 都在其后），
        # 先确保根 logger 挂上 console，否则本条 INFO 会被静默丢弃
        from utils.logger import _ensure_root_logger
        _ensure_root_logger()
        log.info(f'configs/common.yaml: CUDA_VISIBLE_DEVICES='
                 f'{os.environ["CUDA_VISIBLE_DEVICES"]}')


def set_seed(seed, use_gpu):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_gpu:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _snapshot_path(exp_dir, run_name=None):
    """快照路径：带子运行时读 run 级快照（各折配置正确），否则读顶层"""
    return os.path.join(exp_dir, run_name, 'config.yaml') if run_name \
        else os.path.join(exp_dir, 'config.yaml')


def _check_snapshot_consistency(snapshot, args):
    """--dataset/--method 与实验快照一致性校验（省略则跳过，给了且不一致直接报错）"""
    snapshot_method = getattr(snapshot, 'method_config', snapshot.method.name)
    snapshot_dataset = getattr(snapshot, 'dataset_config', snapshot.dataset.name)
    if args.method and args.method != snapshot_method:
        raise SystemExit(f'--method {args.method} 与实验快照的 method '
                         f'{snapshot_method} 不一致，请检查')
    if args.dataset and args.dataset != snapshot_dataset:
        raise SystemExit(f'--dataset {args.dataset} 与实验快照的 dataset '
                         f'{snapshot_dataset} 不一致（续训不允许换数据集）')


def run_train(args):
    distributed, device = _init_distributed()
    is_main = not distributed or dist.get_rank() == 0

    if not args.resume:
        if not args.dataset or not args.method:
            raise SystemExit('新训练必须指定 --dataset 与 --method '
                             '（断点续训用 --resume expNN，无需指定）')

    if args.resume:
        # ------------------------------------------------------ 断点续训
        exp_dir = os.path.join(EXP_ROOT, args.resume)
        if not os.path.isdir(exp_dir):
            raise SystemExit(f'实验目录不存在: {exp_dir}')
        snapshot_path = _snapshot_path(exp_dir, args.run)
        if not os.path.exists(snapshot_path):
            raise SystemExit(f'快照不存在: {snapshot_path}')
        config = yaml_to_ns(load_yaml(snapshot_path))
        _check_snapshot_consistency(config, args)

        # --set 覆盖快照配置（如延长 epochs），并更新快照留档
        if args.set:
            apply_overrides(config, args.set)
            if is_main:
                log.info(f'续训配置覆盖: {args.set}')
        if args.label:
            config.dataset.label = args.label

        # 实验目录/快照/run.log 仅主进程写；其余进程不挂 ExperimentLogger
        logger = None
        if is_main:
            logger = ExperimentLogger(EXP_ROOT, args.resume, args.run, mode='append')
            if args.set:
                logger.save_config(config)  # 更新 run 级快照
        ckpt_path = find_latest_ckpt(exp_dir, args.run)
        set_seed(config.method.output.seed, config.method.output.use_gpu)

        train_loader = build_train_loader(config)
        trainer = Trainer(config, logger, device=device)
        trainer.load_checkpoint(ckpt_path, resume=True)
        trainer.train(train_loader)
        if distributed:
            dist.destroy_process_group()
        return

    # ------------------------------------------------------------ 新训练
    config, _, method_yaml_path = load_config(args.dataset, args.method, args.set)
    if args.label:                          # 标签源（ccs/hcs）随快照留档
        config.dataset.label = args.label

    # 实验目录/快照/run.log 仅主进程写；其余进程不挂 ExperimentLogger
    logger = None
    if is_main:
        logger = ExperimentLogger(EXP_ROOT, args.exp, args.run, mode='create')
        # 快照总是写 run 级（子运行如 LOO 各折的 fold 划分各自留档）；
        # 实验首次创建时另复制一份到顶层供 require_exp 匹配
        logger.save_config(config, method_yaml_path)
    if distributed:
        dist.barrier()  # 等主进程建好实验目录再开跑

    set_seed(config.method.output.seed, config.method.output.use_gpu)

    train_loader = build_train_loader(config)
    trainer = Trainer(config, logger, device=device)
    trainer.train(train_loader)
    if distributed:
        dist.destroy_process_group()


def run_test(args):
    if not args.exp:
        raise SystemExit('测试模式必须指定 --exp，如 --exp exp00')
    exp_dir = os.path.join(EXP_ROOT, args.exp)
    if not os.path.isdir(exp_dir):
        raise SystemExit(f'实验目录不存在: {exp_dir}')

    # 以（run 级）实验快照为准（模型结构、训练配置都来自快照，保证可复现）
    snapshot_path = _snapshot_path(exp_dir, args.run)
    if not os.path.exists(snapshot_path):
        raise SystemExit(f'快照不存在: {snapshot_path}')
    snapshot = yaml_to_ns(load_yaml(snapshot_path))

    snapshot_method = getattr(snapshot, 'method_config', snapshot.method.name)
    snapshot_dataset = getattr(snapshot, 'dataset_config', snapshot.dataset.name)
    if args.method and args.method != snapshot_method:
        raise SystemExit(f'--method {args.method} 与实验 {args.exp} 的 method '
                         f'{snapshot_method} 不一致，请检查')
    # 标签源：快照优先（hcs 训练的模型必须用 hcs 标签评测），--label 显式覆盖
    label = getattr(snapshot.dataset, 'label', None) or args.label
    if args.dataset != snapshot_dataset:
        # 跨数据集评测：用指定数据集的配置替换 dataset 段
        log.info(f'[*] 跨数据集评测: 模型来自 {args.exp}'
                 f'（训练于 {snapshot_dataset}），测试于 {args.dataset}')
        if snapshot.dataset_config.split("/")[-1] == "mpiifacegaze":
            args.run = "all"
        snapshot.dataset, _ = load_dataset_config(args.dataset)
        snapshot.dataset_config = args.dataset
        if args.dataset.split('/')[-1] == "mpiifacegaze":
            snapshot.dataset.split.mode = "all_subjects"
    if label:
        snapshot.dataset.label = label      # ccs 默认不记，hcs 显式留痕于快照
        if label == 'hcs':
            log.info('[*] 标签源: hcs（face_gaze_hcs）')
    if args.set:
        # 测试时的临时覆盖（如 dataset.split.mode=all_subjects），不写回快照
        apply_overrides(snapshot, args.set)
        log.info(f'测试配置覆盖: {args.set}')
    snapshot.experiment = f'{args.dataset}_{snapshot.method.name}'

    if args.ckpt:
        ckpt_dir = os.path.join(exp_dir, args.run, 'ckpt') if args.run \
            else os.path.join(exp_dir, 'ckpt')
        ckpt_path = os.path.join(ckpt_dir, args.ckpt)
        if not os.path.exists(ckpt_path):
            raise SystemExit(f'checkpoint 不存在: {ckpt_path}')
    else:
        # 协议 epoch（CLAUDE.md 约定 8）：从实验快照的 dataset_config 路径
        # 读原始数据集 yaml 的 train.epochs / epochs_by_method，加载
        # epoch_(N-1)_ckpt.pth（恰好训练 N epoch 时即最后一个；旧实验
        # 多训时取协议位而非最新）。--ckpt 显式指定时不覆盖。
        ckpt_path = find_latest_ckpt(exp_dir, args.run)
        try:
            import yaml as _yaml
            snap_file = os.path.join(exp_dir, args.run, 'config.yaml') \
                if args.run else os.path.join(exp_dir, 'config.yaml')
            snap = _yaml.safe_load(open(snap_file))
            ds_cfg_path = snap.get('dataset_config', '')    # 如 zhang2015-insightface/xgaze
            method_name = snap.get('method_config', '')      # 如 resnet50
            if ds_cfg_path:
                ds_yaml = _yaml.safe_load(
                    open(os.path.join('configs/datasets', ds_cfg_path + '.yaml')))
                by_m = (ds_yaml.get('train', {}).get('epochs_by_method') or {})
                protocol_ep = by_m.get(method_name) or \
                    ds_yaml.get('train', {}).get('epochs')
                if protocol_ep:
                    target = os.path.join(
                        exp_dir, args.run or '', 'ckpt',
                        f'epoch_{int(protocol_ep) - 1}_ckpt.pth')
                    if os.path.exists(target):
                        ckpt_path = target
                        log.info(f'协议 epoch={int(protocol_ep)} '
                                 f'({ds_cfg_path})，加载 {os.path.basename(target)}')
                    else:
                        log.warning(f'协议 epoch={protocol_ep} 的 ckpt 不存在，'
                                    f'回退最新 {os.path.basename(ckpt_path)}')
        except Exception as e:
            log.warning(f'协议 epoch 解析失败（{e}），使用最新 ckpt')

    # 测试日志同样追加写入该（子）运行的 run.log，保证溯源完整
    log_dir = os.path.join(exp_dir, args.run, 'logs') if args.run \
        else os.path.join(exp_dir, 'logs')
    attach_file_handler(os.path.join(log_dir, 'run.log'), append=True)

    set_seed(snapshot.method.output.seed, snapshot.method.output.use_gpu)

    test_loader = build_test_loader(snapshot)
    trainer = Trainer(snapshot, logger=None)
    trainer.test(test_loader, ckpt_path)


if __name__ == '__main__':
    _apply_common_env()
    args = parse_args()
    if args.resume and args.test:
        raise SystemExit('--resume 与 --test 不能同时使用')
    if args.resume and args.exp:
        raise SystemExit('--resume 模式下续训输出写入 --resume 的实验目录，'
                         '请勿同时指定 --exp')
    if args.test and not args.dataset:
        raise SystemExit('测试模式必须指定 --dataset（如 --dataset xgaze）')
    if args.test:
        run_test(args)
    else:
        run_train(args)
