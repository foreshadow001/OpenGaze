"""ETH-XGaze 数据集

官方 test 集不公开标注，因此从 80 个被试（官方 train/ 目录）中自行划分：
    train: 75 个被试    test: 5 个被试（subject0106~0111）
划分文件已并入 configs/datasets/xgaze.yaml 的 split 段。
数据现为自预处理版（insightface 管线，/home/hitsz/dataset/xgaze_insightface_224，
h5 在根目录）；其与官方 h5 同为 BGR 存储，加载时统一翻转。
"""
from torch.utils.data import DataLoader

from .base import GazeH5Dataset


def get_train_loader(dataset_config, batch_size, num_workers,
                     sample_size=0, is_shuffle=True):
    split = dataset_config.split.train
    train_set = GazeH5Dataset(
        dataset_path=dataset_config.data_dir,
        sub_folder=split.sub_folder,
        files=split.subjects,
        is_shuffle=is_shuffle,
        sample_size=sample_size,
        bgr_to_rgb=True,
    )
    return DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)


def get_test_loader(dataset_config, batch_size, num_workers, sample_size=0):
    split = dataset_config.split.test
    test_set = GazeH5Dataset(
        dataset_path=dataset_config.data_dir,
        sub_folder=split.sub_folder,
        files=split.subjects,
        is_shuffle=False,
        sample_size=sample_size,
        bgr_to_rgb=True,
    )
    return DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
