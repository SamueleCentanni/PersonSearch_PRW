from collections import OrderedDict

import torch.nn.functional as F
import torchvision
from torch import nn


class MobileNetBackbone(nn.Module):
    """
    MobileNet V3 Large backbone extracting features at stride 16.

    features[0:13] → 112 channels at stride 16
    """

    def __init__(self, mobilenet, split_idx, out_channels):
        super().__init__()
        self.features = nn.Sequential(*list(mobilenet.features[:split_idx]))
        self.out_channels = out_channels

    def forward(self, x):
        feat = self.features(x)
        return OrderedDict([["feat_res4", feat]])


class MobileNetHead(nn.Sequential):
    """
    MobileNet V3 Large head processing RoI-pooled features (analogous to Res5Head).

    features[13:17] → 960 channels (stride 2 internally)

    Returns both the max-pooled input (feat_res4) and max-pooled output (feat_res5),
    matching the Res5Head interface expected by SeqNet.
    """

    def __init__(self, mobilenet, split_idx, backbone_channels, head_channels):
        layers = list(mobilenet.features[split_idx:])
        super().__init__(OrderedDict([(f"layer_{i}", layer) for i, layer in enumerate(layers)]))
        self.out_channels = [backbone_channels, head_channels]

    def forward(self, x):
        feat = super().forward(x)
        x = F.adaptive_max_pool2d(x, 1)
        feat = F.adaptive_max_pool2d(feat, 1)
        return OrderedDict([["feat_res4", x], ["feat_res5", feat]])


# Architecture reference:
# MobileNet V3 Large features:
#   [0]  ConvBNAct(3→16, s=2)    H/2
#   [1]  InvRes(16→16, s=1)      H/2
#   [2]  InvRes(16→24, s=2)      H/4
#   [3]  InvRes(24→24, s=1)      H/4
#   [4]  InvRes(24→40, s=2)      H/8
#   [5]  InvRes(40→40, s=1)      H/8
#   [6]  InvRes(40→40, s=1)      H/8
#   [7]  InvRes(40→80, s=2)      H/16
#   [8]  InvRes(80→80, s=1)      H/16
#   [9]  InvRes(80→80, s=1)      H/16
#   [10] InvRes(80→80, s=1)      H/16
#   [11] InvRes(80→112, s=1)     H/16
#   [12] InvRes(112→112, s=1)    H/16     ← backbone ends here (112 ch)
#   [13] InvRes(112→160, s=2)    H/32     ← head starts here
#   [14] InvRes(160→160, s=1)    H/32
#   [15] InvRes(160→160, s=1)    H/32
#   [16] ConvBNAct(160→960, s=1) H/32     ← head output (960 ch)
#
_MOBILENET_CONFIGS = {
    "mobilenet_v3_large": {
        "factory": torchvision.models.mobilenet_v3_large,
        "split_idx": 13,
        "backbone_channels": 112,
        "head_channels": 960,
    },
}


def build_mobilenet(name="mobilenet_v3_large", pretrained=True):
    """
    Build MobileNet V3 backbone + head, matching the interface of build_resnet().

    Args:
        name: "mobilenet_v3_large"
        pretrained: whether to use ImageNet pretrained weights

    Returns:
        (MobileNetBackbone, MobileNetHead) with the same interface as (Backbone, Res5Head)
    """
    if name not in _MOBILENET_CONFIGS:
        raise ValueError(f"Unknown MobileNet variant: {name}. Choose from {list(_MOBILENET_CONFIGS.keys())}")

    conf = _MOBILENET_CONFIGS[name]
    weights = "DEFAULT" if pretrained else None
    mobilenet = conf["factory"](weights=weights)

    # freeze stem (ConvBNAct 3→16), analogous to ResNet conv1+bn1 freeze
    for param in mobilenet.features[0].parameters():
        param.requires_grad_(False)

    backbone = MobileNetBackbone(mobilenet, conf["split_idx"], conf["backbone_channels"])
    head = MobileNetHead(mobilenet, conf["split_idx"], conf["backbone_channels"], conf["head_channels"])

    return backbone, head
