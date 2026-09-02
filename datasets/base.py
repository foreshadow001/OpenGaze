"""统一 h5 格式数据集基类

所有数据集预处理为同一格式后由此类读取：
    face_patch : (N, 224, 224, 3) uint8
    face_gaze  : (N, 2) float32/float64，(pitch, yaw) 弧度
目录结构：<data_dir>/<sub_folder>/<file>
"""
import os
import random

import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms

# ImageNet 统计量归一化，与 ETH-XGaze 官方训练管线一致
IMAGENET_NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

default_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),  # 像素值 [0,255] → [0,1]
    IMAGENET_NORMALIZE,
])


class GazeH5Dataset(Dataset):
    """统一 h5 读取（懒加载 + swmr， DataLoader 多 worker 安全）

    Args:
        dataset_path: 数据根目录（如 /media/yanglinxuan/ylx/xgaze_insightface_224）
        sub_folder:   h5 文件所在子目录（如 train）
        files:        h5 文件名列表
        bgr_to_rgb:   官方 ETH-XGaze h5 为 BGR 存储，需翻转；自行预处理的 RGB 数据传 False
        sample_size:  >0 时随机抽取前 N 个样本（shuffle 后截断），0 = 全量
        label_field:  标签字段（face_gaze=CCS 相机系 / face_gaze_hcs=头架系，
                      见 label_field_of；v1 产物仅有 face_gaze）
    """

    def __init__(self, dataset_path, sub_folder, files,
                 transform=default_transform, is_shuffle=True,
                 sample_size=0, bgr_to_rgb=False, label_field='face_gaze'):
        self.dataset_path = dataset_path
        self.sub_folder = sub_folder
        self.selected_files = list(files)
        assert len(self.selected_files) > 0, 'h5 文件列表为空'

        # 建立全局索引 → (文件号, 文件内行号) 映射（只读 face_patch 的 shape，随即关闭）
        self.idx_to_kv = []
        for num_i, file_name in enumerate(self.selected_files):
            file_path = os.path.join(dataset_path, sub_folder, file_name)
            with h5py.File(file_path, 'r', swmr=True) as f:
                n = f['face_patch'].shape[0]
            self.idx_to_kv += [(num_i, i) for i in range(n)]

        if is_shuffle:
            random.shuffle(self.idx_to_kv)
        if 0 < sample_size < len(self.idx_to_kv):
            self.idx_to_kv = self.idx_to_kv[:sample_size]

        self.transform = transform
        self.bgr_to_rgb = bgr_to_rgb
        self.label_field = label_field
        self.hdf = None

    def __len__(self):
        return len(self.idx_to_kv)

    def __getitem__(self, idx):
        key, idx_in_file = self.idx_to_kv[idx]
        file_path = os.path.join(self.dataset_path, self.sub_folder,
                                 self.selected_files[key])
        self.hdf = h5py.File(file_path, 'r', swmr=True)

        image = self.hdf['face_patch'][idx_in_file, :]
        if self.bgr_to_rgb:
            image = image[:, :, [2, 1, 0]]
        image = self.transform(image)

        gaze_label = self.hdf[self.label_field][idx_in_file, :].astype('float32')
        self.hdf.close()
        return image, gaze_label


def label_field_of(dataset_config):
    """dataset.label（'ccs' | 'hcs'）→ h5 标签字段名（--label，main.py）"""
    label = getattr(dataset_config, 'label', 'ccs') or 'ccs'
    if label not in ('ccs', 'hcs'):
        raise ValueError(f"未知 dataset.label: {label}（可选 ccs / hcs）")
    return 'face_gaze_hcs' if label == 'hcs' else 'face_gaze'


def make_train_loader(dataset, batch_size, num_workers, distributed=False):
    """训练 DataLoader 统一构造：DDP 时挂 DistributedSampler（按 epoch 重洗牌）"""
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=True)
        return DataLoader(dataset, batch_size=batch_size,
                          num_workers=num_workers, sampler=sampler)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
