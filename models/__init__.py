"""模型工厂：按 method yaml 的 model.backbone 构建网络"""
from .gaze_net import GazeNet, build_model

__all__ = ['GazeNet', 'build_model']
