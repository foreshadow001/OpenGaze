"""从预处理产物抽取面部特征点 + 索引信息 -> <原始数据集目录>/landmarks/（每人/session 一个 h5）

用途：landmarks h5 是"帧清单 + 已提取特征点"的轻量索引（不含 224 图像块，体积 ~1/50），
后续重预处理（如 zhang2015-specific-face-model 逐人模型版）从它开始索引遍历，
跳过 insightface 检测——同帧同特征点，结果逐字节可复现。
landmarks 属"原始数据侧"的派生索引（与原始帧一一对应），故存放在各原始数据集目录下：
  xgaze         /media/hitsz/Expansion/xgaze_raw/data/landmarks/
  mpiifacegaze  /media/hitsz/zyx/MPIIFaceGaze/landmarks/
  gazecapture   /media/hitsz/zyx/GazeCapture/landmarks/
  eve           /media/hitsz/zyx/EVE_dataset/eve_dataset/landmarks/

各数据集保留字段（facial_landmarks_2d + 定位原始帧所需的索引）：
  xgaze         frame_index + cam_index          （帧=subjectNNNN/frameNNNN/camNN.JPG）
  mpiifacegaze  day_index + image_name           （帧=pXX/dayYY/<image_name>）
  gazecapture   frame_index + orientation        （帧=<session>/frames/<frame_index:05d>.jpg）
  eve           frame_index + cam_index + step_index + attrs（steps/cameras 有序列表，
                                                       帧=<steps[si]>/<cameras[ci]>.mp4 #frame_index）
face_gaze / face_mat_norm / face_patch 不复制（重预处理时确定性重算）。

用法（仓库根目录，源/目的地均取自 configs/preprocess/<ds>.yaml 的 output_dir / raw_data_dir）：
  python preprocess/zhang2015-insightface/extract_landmarks.py --dataset eve [--overwrite]
"""
import argparse
import sys
from pathlib import Path

import h5py

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_yaml  # noqa: E402
from utils.logger import get_logger  # noqa: E402

log = get_logger('preprocess.extract_landmarks')

# 数据集 -> (源 h5 相对源根的 glob, 保留字段, 是否复制全部 attrs)
SPECS = {
    'xgaze':         ('subject*.h5',      ['facial_landmarks_2d', 'frame_index', 'cam_index'], False),
    'mpiifacegaze':  ('p*.h5',            ['facial_landmarks_2d', 'day_index', 'image_name'], False),
    'gazecapture':   ('*/*.h5',           ['facial_landmarks_2d', 'frame_index', 'orientation'], False),
    'eve':           ('*/*.h5',           ['facial_landmarks_2d', 'frame_index', 'cam_index', 'step_index'], True),
}


def extract(dataset, src_root, dst_root, overwrite=False):
    pattern, fields, with_attrs = SPECS[dataset]
    src_files = sorted(Path(src_root).glob(pattern))
    if not src_files:
        raise SystemExit(f'{src_root} 下未找到匹配 {pattern} 的 h5')
    out_dir = Path(dst_root) / 'landmarks'
    out_dir.mkdir(exist_ok=True)

    n_done, n_skip, n_rows = 0, 0, 0
    for src in src_files:
        rel = src.relative_to(src_root)
        dst = out_dir / rel
        if dst.is_file() and not overwrite:
            n_skip += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(src, 'r') as fin, h5py.File(dst, 'w') as fout:
            for field in fields:
                if field not in fin:
                    raise KeyError(f'{src} 缺字段 {field}')
                fin.copy(fin[field], fout)      # 保留 dtype/形状（含变长字符串）
            if with_attrs:
                for k, v in fin.attrs.items():
                    fout.attrs[k] = v
        n_done += 1
        n_rows += h5py.File(dst, 'r')['facial_landmarks_2d'].shape[0]
        if n_done % 100 == 0:
            log.info(f'  已抽取 {n_done}/{len(src_files)}')
    log.info(f'{dataset}: 抽取 {n_done} 个（跳过已存在 {n_skip}），'
             f'共 {n_rows} 行特征点 -> {out_dir}')


def main():
    parser = argparse.ArgumentParser(description='抽取特征点+索引到原始数据集 landmarks/')
    parser.add_argument('--dataset', required=True, choices=sorted(SPECS))
    parser.add_argument('--overwrite', action='store_true', help='覆盖已存在的 landmarks h5')
    args = parser.parse_args()

    cfg = load_yaml(str(PROJECT_ROOT / 'configs' / 'preprocess' / f'{args.dataset}.yaml'))
    src_root, dst_root = cfg['output_dir'], cfg['raw_data_dir']
    log.info(f'数据集 {args.dataset} | 源 {src_root} | landmarks -> {dst_root}/landmarks')
    extract(args.dataset, src_root, dst_root, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
