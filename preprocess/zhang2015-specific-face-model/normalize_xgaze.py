"""zhang2015-specific-face-model 版 XGaze 归一化（协议见本目录 normalization_protocol.md）

复用 zhang2015-insightface 的 XGazePreprocessor 骨架（按路径 import，目录含连字符），
仅覆写 `_geometry_for`：逐 (被试, 相机) 查个性化模型——
  <face_model_root>/subject{id:04d}/cam{cc:02d}_model6.txt 存在 → 用之；
  不存在（建模时未过视角准则的相机）→ 回退通用模型，recorder 记 fallback_generic。
仅支持 landmarks 模式（landmarks_dir 必填——本管线不做 insightface 检测）。
pnp_points: 6（默认，D1 决策 A）用 model6+IDX6；28（A/B 对比）用 model28+RIGID，
两种配置下 normalizeData_face 均用 model6 的 6 行（旋转中心几何一致）。

用法：python preprocess.py --dataset xgaze --method zhang2015-specific-face-model
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HERE = os.path.dirname(os.path.abspath(__file__))
# utils 包在仓库根；基类模块按路径加载（zhang2015-insightface 目录含连字符）
sys.path.insert(0, _PROJECT)
_base_path = Path(_HERE).parent / 'zhang2015-insightface' / 'normalize_xgaze.py'
_spec = importlib.util.spec_from_file_location('z15if_normalize_xgaze', _base_path)
_z15if = importlib.util.module_from_spec(_spec)
sys.modules['z15if_normalize_xgaze'] = _z15if
_spec.loader.exec_module(_z15if)

# 28 点刚性核心（与 get_face_model/face_model_core.RIGID 一致，此处独立列出避免跨目录 import）
RIGID = [i for i in list(range(33, 43)) + list(range(72, 87)) + list(range(87, 97))
         if i not in (34, 38, 86, 88, 92, 94, 95)]

from utils.logger import get_logger  # noqa: E402

log = get_logger('preprocess.specific_face_model.xgaze')


class SpecificXGazePreprocessor(_z15if.XGazePreprocessor):
    """XGaze 归一化 + 逐被试逐相机个性化人脸模型"""

    def __init__(self, config):
        super().__init__(config)
        if not self.landmarks_dir:
            raise SystemExit('zhang2015-specific-face-model 管线要求 landmarks_dir '
                             '（landmarks 模式，不做 insightface 检测）')
        self.face_model_root = config.face_model_root
        self.pnp_points = int(getattr(config, 'pnp_points', 6))
        if self.pnp_points not in (6, 28):
            raise SystemExit(f'pnp_points 只支持 6 / 28，配置为 {self.pnp_points}')
        self._model_cache = {}
        self.n_specific = self.n_fallback = 0

    def _geometry_for(self, subject_index, cam):
        key = (subject_index, cam)
        if key not in self._model_cache:
            sub = Path(self.face_model_root) / f'subject{subject_index:04d}'
            p6 = sub / f'cam{cam:02d}_model6.txt'
            if p6.is_file():
                m6 = np.loadtxt(p6).astype(float)
                if self.pnp_points == 28:
                    p28 = sub / f'cam{cam:02d}_model28.txt'
                    if p28.is_file():
                        self._model_cache[key] = (np.loadtxt(p28).astype(float),
                                                  m6, RIGID)
                    else:                       # 28 点模型缺失，退 6 点
                        self._model_cache[key] = (m6, m6, self.IDX6)
                else:
                    self._model_cache[key] = (m6, m6, self.IDX6)
                self.n_specific += 1
            else:                               # 建模未覆盖的相机：回退通用
                self._model_cache[key] = (self.face_model_use,
                                          self.face_model_use, self.IDX6)
                self.n_fallback += 1
                # recorder 在 process_subject 作用域内逐帧可记；此处按 (人,相机) 汇总一次
                self._fallback_keys = getattr(self, '_fallback_keys', set())
                self._fallback_keys.add(key)
        return self._model_cache[key]

    def run(self, recorder):
        total = super().run(recorder)
        for subject_index, cam in getattr(self, '_fallback_keys', set()):
            recorder.add(subject_index, f'cam{cam:02d}', 'fallback_generic')
        log.info(f'几何来源: 个性化 {(self.n_specific)} 组 / 回退通用 {self.n_fallback} 组')
        return total


def run(config, recorder):
    """preprocess.py 入口适配"""
    return SpecificXGazePreprocessor(config).run(recorder)
