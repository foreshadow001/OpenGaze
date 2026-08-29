"""zhang2015-specific-face-model 版 EVE 归一化（协议见本目录 normalization_protocol.md）

复用 zhang2015-insightface 的 EVEPreprocessor（包按路径 import），覆写 `_model_for`：
逐 (被试, 相机) 查个性化模型——
  <face_model_root>/<被试>/cam{cc:02d}_model6.txt 存在 → 用之；该相机缺失（未过建模
  视角准则）→ 回退通用（记 fallback_generic）。
仅支持 landmarks 模式。EVE 44 被试全部有模型（部分被试个别相机除外）。

用法：python preprocess.py --dataset eve --method zhang2015-specific-face-model
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT)
_pkg_init = Path(_HERE).parent / 'zhang2015-insightface' / 'eve' / '__init__.py'
_spec = importlib.util.spec_from_file_location(
    'z15if_eve', _pkg_init, submodule_search_locations=[str(_pkg_init.parent)])
_z15eve = importlib.util.module_from_spec(_spec)
sys.modules['z15if_eve'] = _z15eve
_spec.loader.exec_module(_z15eve)

from utils.logger import get_logger  # noqa: E402

log = get_logger('preprocess.specific_face_model.eve')


class SpecificEVEPreprocessor(_z15eve.EVEPreprocessor):
    """EVE 归一化 + 逐被试×相机个性化人脸模型"""

    def __init__(self, config):
        super().__init__(config)
        if not self.landmarks_dir:
            raise SystemExit('zhang2015-specific-face-model 管线要求 landmarks_dir '
                             '（landmarks 模式，不做 insightface 检测）')
        self.face_model_root = config.face_model_root
        self._model_cache = {}
        self._fallback_keys = set()

    def _model_for(self, subject, cam_index):
        key = (subject, int(cam_index))
        if key not in self._model_cache:
            p = Path(self.face_model_root) / subject / f'cam{int(cam_index):02d}_model6.txt'
            if p.is_file():
                self._model_cache[key] = np.loadtxt(p).astype(float)
            else:
                self._model_cache[key] = self.face_model_use
                self._fallback_keys.add(key)
        return self._model_cache[key]

    def run(self, recorder):
        total = super().run(recorder)
        for s, c in sorted(self._fallback_keys):
            recorder.add(s, f'cam{c:02d}', 'fallback_generic')
        log.info(f'几何来源: 个性化 {(len(self._model_cache) - len(self._fallback_keys))} 组 / '
                 f'回退通用 {len(self._fallback_keys)} 组')
        return total


def run(config, recorder):
    """preprocess.py 入口适配"""
    return SpecificEVEPreprocessor(config).run(recorder)
