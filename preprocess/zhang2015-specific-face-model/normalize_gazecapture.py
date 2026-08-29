"""zhang2015-specific-face-model 版 GazeCapture 归一化（协议见本目录 normalization_protocol.md）

复用 zhang2015-insightface 的 GazeCapturePreprocessor（包按路径 import），覆写
`_model_for`：逐 (session, 朝向) 查个性化模型——
  <face_model_root>/<session>/ori{o}_model6.txt 存在 → 用之；该朝向缺失 → 回退通用
（记 fallback_generic）。
仅支持 landmarks 模式；**session 范围 = configs/splits/gazecapture_sfm.yaml**
（FAZE 筛选 ∩ 建模成功，2026-08-27 定稿：建模失败的 session 不预处理不训练测试）。

用法：python preprocess.py --dataset gazecapture --method zhang2015-specific-face-model
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT)
_pkg_init = Path(_HERE).parent / 'zhang2015-insightface' / 'gazecapture' / '__init__.py'
_spec = importlib.util.spec_from_file_location(
    'z15if_gazecapture', _pkg_init,
    submodule_search_locations=[str(_pkg_init.parent)])
_z15gc = importlib.util.module_from_spec(_spec)
sys.modules['z15if_gazecapture'] = _z15gc
_spec.loader.exec_module(_z15gc)

from utils.logger import get_logger  # noqa: E402

log = get_logger('preprocess.specific_face_model.gazecapture')


class SpecificGazeCapturePreprocessor(_z15gc.GazeCapturePreprocessor):
    """GazeCapture 归一化 + 逐 session×朝向个性化人脸模型"""

    def __init__(self, config):
        super().__init__(config)
        if not self.landmarks_dir:
            raise SystemExit('zhang2015-specific-face-model 管线要求 landmarks_dir '
                             '（landmarks 模式，不做 insightface 检测）')
        self.face_model_root = config.face_model_root
        self._model_cache = {}
        self._fallback_keys = set()

    def _model_for(self, session, ori):
        key = (session, int(ori))
        if key not in self._model_cache:
            p = Path(self.face_model_root) / session / f'ori{int(ori)}_model6.txt'
            if p.is_file():
                self._model_cache[key] = np.loadtxt(p).astype(float)
            else:
                self._model_cache[key] = self.face_model_use
                self._fallback_keys.add(key)
        return self._model_cache[key]

    def run(self, recorder):
        total = super().run(recorder)
        for s, o in sorted(self._fallback_keys):
            recorder.add(s, f'ori{o}', 'fallback_generic')
        log.info(f'几何来源: 个性化 {(len(self._model_cache) - len(self._fallback_keys))} 组 / '
                 f'回退通用 {len(self._fallback_keys)} 组')
        return total


def run(config, recorder):
    """preprocess.py 入口适配"""
    return SpecificGazeCapturePreprocessor(config).run(recorder)
