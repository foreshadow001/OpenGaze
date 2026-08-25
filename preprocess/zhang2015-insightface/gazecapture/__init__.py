"""GazeCapture 预处理器（zhang2015-insightface 管线）

实现见 preprocessor.py（GazeCapturePreprocessor）；本文件仅做包出口，
供 preprocess.py 的目录形式加载（需暴露模块级 run(config, recorder)）。
"""
from .preprocessor import (  # noqa: F401
    GazeCapturePreprocessor,
    run,
    _dot_to_ccs_mm,
    _gaze_point_cam,
    LANDSCAPE_SIGN,
)
