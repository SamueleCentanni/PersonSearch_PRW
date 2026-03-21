from collections import OrderedDict

import torch.nn.functional as F
import torchvision
from torch import nn


class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt backbone extracting features at stride 16.

    Takes features[0:6] (stem + stages 1-3), producing stride-16 feature maps.
    """

    def __init__(self, convnext, out_channels):
        super().__init__()
        self.features = nn.Sequential(*list(convnext.features[:6]))
        self.out_channels = out_channels

    def forward(self, x):
        feat = self.features(x)
        return OrderedDict([["feat_res4", feat]])


class ConvNeXtHead(nn.Sequential):
    """
    ConvNeXt head processing RoI-pooled features (analogous to Res5Head).

    Takes features[6:8] (downsample + stage 4), producing stride-32 features.
    Returns both max-pooled input (feat_res4) and max-pooled output (feat_res5).
    """

    def __init__(self, convnext, backbone_channels, head_channels):
        layers = list(convnext.features[6:])
        super().__init__(OrderedDict([(f"layer_{i}", layer) for i, layer in enumerate(layers)]))
        self.out_channels = [backbone_channels, head_channels]

    def forward(self, x):
        feat = super().forward(x)
        x = F.adaptive_max_pool2d(x, 1)
        feat = F.adaptive_max_pool2d(feat, 1)
        return OrderedDict([["feat_res4", x], ["feat_res5", feat]])


# Architecture reference (ConvNeXt features):
#   [0] stem:   Conv2d(3→C1, 4×4, stride=4) + LayerNorm    → stride 4
#   [1] stage1: CNBlocks (C1)                                → stride 4
#   [2] down:   LayerNorm + Conv2d(C1→C2, 2×2, stride=2)    → stride 8
#   [3] stage2: CNBlocks (C2)                                → stride 8
#   [4] down:   LayerNorm + Conv2d(C2→C3, 2×2, stride=2)    → stride 16
#   [5] stage3: CNBlocks (C3)                                → stride 16  ← backbone ends
#   [6] down:   LayerNorm + Conv2d(C3→C4, 2×2, stride=2)    → stride 32  ← head starts
#   [7] stage4: CNBlocks (C4)                                → stride 32  ← head output
#
# Channel configs:
#   small: C1=96,  C2=192, C3=384, C4=768

_CONVNEXT_CONFIGS = {
    "convnext_small": {
        "factory": torchvision.models.convnext_small,
        "backbone_channels": 384,
        "head_channels": 768,
    },
}


def build_convnext(name="convnext_tiny", pretrained=True):
    """
    Build ConvNeXt backbone + head, matching the interface of build_resnet().

    Args:
        name: "convnext_small"
        pretrained: whether to use ImageNet pretrained weights

    Returns:
        (ConvNeXtBackbone, ConvNeXtHead) with the same interface as (Backbone, Res5Head)
    """
    if name not in _CONVNEXT_CONFIGS:
        raise ValueError(f"Unknown ConvNeXt variant: {name}. Choose from {list(_CONVNEXT_CONFIGS.keys())}")

    conf = _CONVNEXT_CONFIGS[name]
    weights = "DEFAULT" if pretrained else None
    convnext = conf["factory"](weights=weights)

    # freeze stem (Conv2d 3→C1 + LayerNorm), analogous to ResNet conv1+bn1 freeze
    for param in convnext.features[0].parameters():
        param.requires_grad_(False)

    backbone = ConvNeXtBackbone(convnext, conf["backbone_channels"])
    head = ConvNeXtHead(convnext, conf["backbone_channels"], conf["head_channels"])

    return backbone, head
