"""重打 v3 产物 face_patch 的 h5 chunk：(50,224,224,3) → (1,224,224,3)

背景（2026-09-03 诊断）：v3 四套 `*_noroll_224` 的 face_patch 沿用了
v2-xgaze 的 BATCH_SIZE=50 大 chunk——训练 shuffle 随机读时，取 1 行要
读出并 lzf 解压整块 7.5MB（放大 50 倍），4 卡 4090 + NVMe 只能跑到
3 batch/s（lzf 单核解压 ~250MB/s 打满 20 个 dataloader worker）。
逐样本 chunk 后恢复 ~10 batch/s（v1/v2-eve 同款布局的实测水平）。

做法：顺序读源文件（整 chunk 解压一次，顺序 I/O）→ 逐样本 chunk 重写
face_patch（保留 lzf），其余数据集/attrs 原样拷贝（chunk 改为逐行）。
幂等可续跑：目标文件已存在且 chunk/行数正确则跳过；写 .tmp 后原子改名，
验证若干行与源一致才落位。

用法（仓库根目录）:
    python preprocess/ours-without-roll/repack_chunks.py                # 默认 12 进程
    python preprocess/ours-without-roll/repack_chunks.py --workers 16
    python preprocess/ours-without-roll/repack_chunks.py --root /media/yanglinxuan/sfm   # 重打 sfm 正本
"""
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

_PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT))
from utils.logger import get_logger

log = get_logger('preprocess.ours_without_roll.repack')

ROOT = Path('/data/others_preprocessed_datasets')
DIRS = ['xgaze_noroll_224', 'eve_noroll_224',
        'gazecapture_noroll_224', 'mpiifacegaze_noroll_224']
BLOCK = 512          # 每次 copy 的行数（512×150KB ≈ 77MB）


def _copy_ds(src, dst, name, lzf=False):
    s = src[name]
    n = s.shape[0]
    chunks = (1,) + tuple(s.shape[1:]) if s.ndim > 1 else (min(n, 4096),)
    kw = dict(chunks=chunks, maxshape=s.maxshape, dtype=s.dtype)
    if lzf:
        kw['compression'] = 'lzf'
    d = dst.create_dataset(name, shape=s.shape, **kw)
    for i in range(0, n, BLOCK):
        d[i:i + BLOCK] = s[i:i + BLOCK]
    return d


def repack_one(path: Path) -> tuple:
    """重打单个 h5（幂等）；返回 (状态, 文件名, 行数)"""
    tmp = path.with_suffix(path.suffix + '.rechunk_tmp.h5')
    # 已重打完：chunk 逐样本 → 跳过（0 行文件也算完成，否则死循环）
    if path.is_file():
        try:
            with h5py.File(path, 'r') as f:
                if f['face_patch'].chunks == (1, 224, 224, 3):
                    return 'skip', path.name, f['face_patch'].shape[0]
        except OSError:
            pass                        # 损坏/半截 → 重做
    with h5py.File(path, 'r') as src:
        n = src['face_patch'].shape[0]
        with h5py.File(tmp, 'w') as dst:
            dst.attrs.update({k: v for k, v in src.attrs.items()})
            for k in src.keys():
                _copy_ds(src, dst, k, lzf=(k == 'face_patch'))
        # 验证：形状 + 首/中/尾行 + 全部标签逐位（0 行文件只验形状）
        with h5py.File(tmp, 'r') as dst:
            assert dst['face_patch'].shape == (n, 224, 224, 3)
            for i in (sorted({0, n // 2, n - 1}) if n > 0 else ()):
                assert np.array_equal(dst['face_patch'][i], src['face_patch'][i])
            if n > 0:
                for k in ('face_gaze', 'face_gaze_hcs', 'face_head_elev_azim'):
                    if k in src:
                        assert np.array_equal(dst[k][:], src[k][:])
    os.replace(tmp, path)
    return 'done', path.name, n


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int,
                    default=min(12, (os.cpu_count() or 8)))
    ap.add_argument('--root', type=str, default=str(ROOT))
    ap.add_argument('--dirs', nargs='+', default=DIRS)
    args = ap.parse_args()

    files = sorted(p for d in args.dirs
                   for p in (Path(args.root) / d).rglob('*.h5'))
    todo = [p for p in files if not p.name.endswith('.rechunk_tmp.h5')]
    total_gb = sum(p.stat().st_size for p in todo) / 1e9
    log.info(f'重打 chunk：{len(todo)} 个 h5 / {total_gb:.0f} GB，'
             f'{args.workers} 进程，目标 chunks=(1,224,224,3)')
    t0 = time.time()
    done = skip = 0
    with Pool(args.workers) as pool, tqdm(total=len(todo), unit='file',
                                          desc='repack') as bar:
        for status, name, n in pool.imap_unordered(repack_one, todo):
            if status == 'skip':
                skip += 1
            else:
                done += 1
            bar.set_postfix({'done': done, 'skip': skip, 'rows': n})
            bar.update(1)
    dt = time.time() - t0
    log.info(f'完成：重打 {done} / 跳过 {skip} / 共 {len(todo)}，'
             f'{dt / 60:.1f} min（{total_gb / max(dt, 1):.0f} MB/s）')


if __name__ == '__main__':
    main()
