"""zhang2015-specific-face-model 版 MPIIFaceGaze 归一化（协议见本目录 normalization_protocol.md）

复用 zhang2015-insightface 的 normalize_mpiifacegaze.py（函数式结构，按路径 import），
逐被试把模块级 FACE_MODEL_USE 换成个性化模型：
  <face_model_root>/pXX/cam00_model6.txt 存在 → 用之；缺失 → 保持通用（记 fallback_generic）。
仅支持 landmarks 模式（landmarks_dir 必填）。单人单相机，D1 决策：PnP 固定 6 点（IDX6）。

用法：python preprocess.py --dataset mpiifacegaze --method zhang2015-specific-face-model
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT)
_base_path = Path(_HERE).parent / 'zhang2015-insightface' / 'normalize_mpiifacegaze.py'
_spec = importlib.util.spec_from_file_location('z15if_normalize_mpii', _base_path)
mod = importlib.util.module_from_spec(_spec)
sys.modules['z15if_normalize_mpii'] = mod
_spec.loader.exec_module(mod)
GENERIC = mod.FACE_MODEL_USE               # 通用 6 点模型（回退用）

from utils.logger import get_logger  # noqa: E402

log = get_logger('preprocess.specific_face_model.mpiifacegaze')


def run(config, recorder):
    """逐被试：换模型 -> 调 insightface 版 process_subject（landmarks 模式）"""
    if not (getattr(config, 'landmarks_dir', '') or None):
        raise SystemExit('zhang2015-specific-face-model 管线要求 landmarks_dir '
                         '（landmarks 模式，不做 insightface 检测）')
    from preprocess.common import FailureRecorder  # noqa: F401 （类型引用）

    fmr = Path(config.face_model_root)
    subjects = config.subjects if config.subjects else sorted(
        p.name[:-3] for p in Path(config.landmarks_dir).glob('p*.h5'))
    log.info(f'MPII specific: {len(subjects)} 人 | 模型根 {fmr} | 输出 {config.output_dir}')
    total, n_specific, n_fallback = 0, 0, 0
    t0 = __import__('time').time()
    for subject in subjects:
        p = fmr / subject / 'cam00_model6.txt'
        if p.is_file():
            mod.FACE_MODEL_USE = np.loadtxt(p).astype(float)
            n_specific += 1
        else:
            mod.FACE_MODEL_USE = GENERIC
            n_fallback += 1
            recorder.add(subject, 'cam00', 'fallback_generic')
            log.warning(f'{subject}: 无专属模型, 回退通用')
        total += mod.process_subject(subject, None, config, recorder)
    log.info(f'全部完成: {len(subjects)} 人, {total} 样本 | 个性化 {n_specific} / '
             f'回退 {n_fallback} | 耗时 {(__import__("time").time() - t0) / 60:.1f} 分钟')
    return total
