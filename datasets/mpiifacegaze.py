"""MPIIFaceGaze 数据集（已预处理为统一 h5，见 ~/data-preprocessing-gaze/normalize_mpiifacegaze_h5.py）

15 个被试（p00~p14），h5 直接位于 data_dir 下（无子目录），BGR 存储（与 xgaze 相同，
预处理管线同构：insightface PnP 归一化，face_gaze 为 (theta, phi) 弧度，约定与官方一致）。

split 模式：
    leave_one_out  折 i 以 p{i:02d} 为测试被试，其余 14 人训练（within-dataset 协议）
    all_subjects   15 人全量，train = test = 全部（cross-dataset 的源训练 / 目标测试）
"""
from torch.utils.data import DataLoader

from .base import GazeH5Dataset, make_train_loader


def _split_subjects(split):
    """按配置的 split 段返回 (train_subjects, test_subjects)"""
    subjects = list(split.subjects)
    mode = split.mode
    if mode == 'leave_one_out':
        fold = split.fold
        if not 0 <= fold < len(subjects):
            raise ValueError(f'fold {fold} 超出范围 0~{len(subjects) - 1}')
        test_subjects = [subjects[fold]]
        train_subjects = [s for i, s in enumerate(subjects) if i != fold]
    elif mode == 'all_subjects':
        train_subjects = subjects
        test_subjects = subjects
    else:
        raise NotImplementedError(
            f'未知 split.mode: {mode}，可选 leave_one_out / all_subjects')
    return train_subjects, test_subjects


def get_train_loader(dataset_config, batch_size, num_workers,
                     sample_size=0, is_shuffle=True, distributed=False):
    train_subjects, _ = _split_subjects(dataset_config.split)
    split = dataset_config.split
    train_set = GazeH5Dataset(
        dataset_path=dataset_config.data_dir,
        sub_folder=split.sub_folder,
        files=train_subjects,
        is_shuffle=is_shuffle,
        sample_size=sample_size,
        bgr_to_rgb=True,   # 预处理管线与 xgaze 同构，BGR 存储
    )
    return make_train_loader(train_set, batch_size, num_workers, distributed)


def get_test_loader(dataset_config, batch_size, num_workers, sample_size=0):
    _, test_subjects = _split_subjects(dataset_config.split)
    split = dataset_config.split
    test_set = GazeH5Dataset(
        dataset_path=dataset_config.data_dir,
        sub_folder=split.sub_folder,
        files=test_subjects,
        is_shuffle=False,
        sample_size=sample_size,
        bgr_to_rgb=True,
    )
    return DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
