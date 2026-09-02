"""GazeCapture 数据集

官方划分为 session 级（train 1321 含官方 val / test 150，excluded 3），
session 列表与预处理共用 configs/splits/gazecapture_sessions.yaml——loader 按数据集
配置 split 段的 split_file + split_key 读取，h5 为 <data_dir>/<sub_folder>/<session>.h5，
BGR 存储（平台统一约定），加载时翻转。session 数量大且跨设备/被试，不做 LOO。
"""
import os

from torch.utils.data import DataLoader

from .base import GazeH5Dataset, label_field_of, make_train_loader
from utils.config import PROJECT_ROOT, load_yaml


def _session_files(split):
    """split 段 → session h5 文件列表（官方 session 划分，见 configs/splits/）"""
    split_file = getattr(split, 'split_file', '') or \
        'configs/splits/gazecapture_sessions.yaml'
    split_key = getattr(split, 'split_key', '') or 'train'
    sessions = load_yaml(os.path.join(PROJECT_ROOT, split_file))[split_key]
    return [f'{s}.h5' for s in sessions]


def get_train_loader(dataset_config, batch_size, num_workers,
                     sample_size=0, is_shuffle=True, distributed=False):
    split = dataset_config.split.train
    train_set = GazeH5Dataset(
        dataset_path=dataset_config.data_dir,
        sub_folder=split.sub_folder,
        files=_session_files(split),
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
        files=_session_files(split),
        is_shuffle=False,
        sample_size=sample_size,
        bgr_to_rgb=True,
        label_field=label_field_of(dataset_config),
    )
    return DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
