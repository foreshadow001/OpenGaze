"""视线估计网络：backbone + 线性回归头，输出 (pitch, yaw)

与 ETH-XGaze 官方 model.py 结构一致（backbone 池化特征 → Linear(feat_dim, 2)），
预训练权重可对齐。
"""
import torch.nn as nn

from .resnet import resnet18, resnet50

# backbone → (构造函数, 池化特征维度)
BACKBONES = {
    'resnet18': (resnet18, 512),
    'resnet50': (resnet50, 2048),
}


class GazeNet(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True, num_out=2):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f'未知 backbone: {backbone}，可选: {list(BACKBONES)}')
        backbone_fn, feat_dim = BACKBONES[backbone]
        self.backbone = backbone_fn(pretrained=pretrained)
        self.gaze_fc = nn.Linear(feat_dim, num_out)

    def forward(self, x):
        feature = self.backbone(x)
        feature = feature.view(feature.size(0), -1)
        return self.gaze_fc(feature)


def build_model(model_config):
    """按 method yaml 的 model 段构建网络"""
    return GazeNet(
        backbone=getattr(model_config, 'backbone', 'resnet50'),
        pretrained=getattr(model_config, 'pretrained', True),
        num_out=getattr(model_config, 'num_out', 2),
    )
