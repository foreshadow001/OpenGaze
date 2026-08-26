"""数据集工厂：按 config.dataset.name 分发到对应模块

新增数据集：在 datasets/ 下加一个模块（提供 get_train_loader / get_test_loader），
并注册到 DATASETS。
"""
import torch.distributed as dist

from . import eve, gazecapture, mpiifacegaze, xgaze

DATASETS = {
    'xgaze': xgaze,
    'mpiifacegaze': mpiifacegaze,
    'eve': eve,
    'gazecapture': gazecapture,
}


def build_train_loader(config):
    dataset = config.dataset
    loader_cfg = dataset.dataloader
    # torchrun 多卡训练：batch_size 语义为「全局批大小」，按卡数均分到各 rank
    # （余数丢弃，全局有效批 = floor(batch_size/world)*world；单卡不受影响）
    distributed = dist.is_available() and dist.is_initialized()
    batch_size = config.method.train.batch_size
    if distributed:
        batch_size = max(1, batch_size // dist.get_world_size())
    return DATASETS[dataset.name].get_train_loader(
        dataset,
        batch_size=batch_size,
        num_workers=loader_cfg.num_workers,
        sample_size=loader_cfg.train_sample_size,
        is_shuffle=True,
        distributed=distributed,
    )


def build_test_loader(config):
    dataset = config.dataset
    loader_cfg = dataset.dataloader
    return DATASETS[dataset.name].get_test_loader(
        dataset,
        batch_size=config.method.train.batch_size,
        num_workers=loader_cfg.num_workers,
        sample_size=getattr(loader_cfg, 'test_sample_size', 0),
    )
