"""数据集工厂：按 config.dataset.name 分发到对应模块

新增数据集：在 datasets/ 下加一个模块（提供 get_train_loader / get_test_loader），
并注册到 DATASETS。
"""
from . import mpiifacegaze, xgaze

DATASETS = {
    'xgaze': xgaze,
    'mpiifacegaze': mpiifacegaze,
}


def build_train_loader(config):
    dataset = config.dataset
    loader_cfg = dataset.dataloader
    return DATASETS[dataset.name].get_train_loader(
        dataset,
        batch_size=config.method.train.batch_size,
        num_workers=loader_cfg.num_workers,
        sample_size=loader_cfg.train_sample_size,
        is_shuffle=True,
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
