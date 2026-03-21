import numpy as np
from PIL import Image
import torch


class BaseDataset:
    """
    Base class of person search dataset.
    """

    def __init__(self, root, transforms, split):
        self.root = root
        self.transforms = transforms
        self.split = split
        
        # "train" split is further divided into "train" and "val" splits for training and validation, respectively.
        # "val" is then further split into "val_gallery" and "val_query" splits for evaluation, to emulate the test setting.
        assert self.split in ("train", "gallery", "query", "val_gallery", "val_query")
        self.annotations = self._load_annotations()

    def _load_annotations(self):
        """
        For each image, load its annotation that is a dictionary with the following keys:
            img_name (str): image name
            img_path (str): image path
            boxes (np.array[N, 4]): ground-truth boxes in (x1, y1, x2, y2) format
            pids (np.array[N]): person IDs corresponding to these boxes
            cam_id (int): camera ID (only for PRW dataset)
        """
        raise NotImplementedError

    def __getitem__(self, index):
        """
        Given the index of an image, return the image and its corresponding target, which is a dictionary with the following keys
            img_name (str): image name
            boxes (torch.Tensor[N, 4]): ground-truth boxes in (x1, y1, x2, y2) format
            labels (torch.Tensor[N]): person IDs corresponding to these boxes
        """
        
        anno = self.annotations[index]
        img = Image.open(anno["img_path"]).convert("RGB")
        boxes = torch.as_tensor(anno["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(anno["pids"], dtype=torch.int64)
        target = {"img_name": anno["img_name"], "boxes": boxes, "labels": labels}
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.annotations)