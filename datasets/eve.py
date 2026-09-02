"""EVE 数据集

平台划分（官方 test01–06 无 Tobii 标注弃用）：train01–39 训练 / val01–05 测试，
划分列表已并入 configs/datasets/zhang2015-insightface/eve.yaml 的 split 段（与 configs/preprocess/zhang2015-insightface/eve.yaml
一致；勘探与决策记录见 preprocess/zhang2015-insightface/eve/dataset_report.md）。
自预处理 h5 为 BGR 存储（平台统一约定，见 CLAUDE.md 约定 5），加载时翻转。
"""
from torch.utils.data import DataLoader

from .base import GazeH5Dataset, label_field_of, make_train_loader


def get_train_loader(dataset_config, batch_size, num_workers,
                     sample_size=0, is_shuffle=True, distributed=False):
    split = dataset_config.split.train
    train_set = GazeH5Dataset(
        dataset_path=dataset_config.data_dir,
        sub_folder=split.sub_folder,
        files=split.subjects,
        is_shuffle=is_shuffle,
        sample_size=sample_size,
        bgr_to_rgb=True,
        label_field=label_field_of(dataset_config),
    )
    return make_train_loader(train_set, batch_size, num_workers, distributed)


def get_test_loader(dataset_config, batch_size, num_workers, sample_size=0):
    split = dataset_config.split.test
    test_set = GazeH5Dataset(
        dataset_path=dataset_config.data_dir,
        sub_folder=split.sub_folder,
        files=split.subjects,
        is_shuffle=False,
        sample_size=sample_size,
        bgr_to_rgb=True,
        label_field=label_field_of(dataset_config),
    )
    return DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
