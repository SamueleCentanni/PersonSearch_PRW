from collections import OrderedDict

import torch.nn.functional as F
import torchvision
from torch import nn


# from the paper: first part is faster R-CNN backbone (ResNet-50 pretrained on ImageNet) followed by res5 head. At the end it outputs Cls and Boxes.
# The res1 - res4 are taken to extract the 1024-channel stem feature maps of the image.
# Second part instead is baseline (from NAE paper, Chen er al. 2020). The boxes are fed into res5 to extract 2048-dim features, which are then mapped to 256-dim. It uses these 2048-dim features
# to calculate regressors and 256-dim features to perform classification and re-ID tasks.

class Backbone(nn.Sequential):
    def __init__(self, resnet):
        super(Backbone, self).__init__(
            OrderedDict(
                [
                    ["conv1", resnet.conv1],
                    ["bn1", resnet.bn1],
                    ["relu", resnet.relu],
                    ["maxpool", resnet.maxpool],
                    ["layer1", resnet.layer1],  # res2
                    ["layer2", resnet.layer2],  # res3
                    ["layer3", resnet.layer3],  # res4
                ]
            )
        )
        self.out_channels = 1024

    def forward(self, x):
        # using the forward method from nn.Sequential
        feat = super(Backbone, self).forward(x)
        return OrderedDict([["feat_res4", feat]])


class Res5Head(nn.Sequential):
    def __init__(self, resnet):
        super(Res5Head, self).__init__(OrderedDict([["layer4", resnet.layer4]]))  # res5
        self.out_channels = [1024, 2048]

    def forward(self, x):
        feat = super(Res5Head, self).forward(x)
        x = F.adaptive_max_pool2d(x, 1)
        feat = F.adaptive_max_pool2d(feat, 1)
        return OrderedDict([["feat_res4", x], ["feat_res5", feat]])


def build_resnet(name="resnet50", pretrained=True):
    weights = "DEFAULT" if pretrained else None # DEFAUL = pretrained weights on ImageNet
    resnet = torchvision.models.resnet.__dict__[name](weights=weights)

    # freeze layers
    resnet.conv1.weight.requires_grad_(False)
    resnet.bn1.weight.requires_grad_(False)
    resnet.bn1.bias.requires_grad_(False)

    return Backbone(resnet), Res5Head(resnet)