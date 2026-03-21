from __future__ import annotations

import random
from torchvision.transforms import functional as TF


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = TF.to_tensor(image)
        return image, target


def build_transforms(is_train: bool):
    transforms = [ToTensor()]
    if is_train:
        transforms.append(RandomHorizontalFlip())
    return Compose(transforms)
