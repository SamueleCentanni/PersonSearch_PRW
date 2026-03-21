from __future__ import annotations

from copy import deepcopy
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from dataset.build import build_dataset
from dataset.transforms import build_transforms


def build_prw_train_val_split(
    dataset_root: str,
    val_size: int = 570,
    seed: int = 0,
):
    """
    Create an internal validation protocol from the PRW train split to create
    train, validation gallery, and validation query datasets.

    Returns train_dataset, val_gallery_dataset, val_query_dataset.
    """
    
    # This is the full train split, which will be split into train and val splits
    # I set is_train=False because in this step I don't want to apply Horizontal Flip, since this would affect the validation split 
    all_train_for_split = build_dataset(
        "PRW", dataset_root, build_transforms(is_train=False), "train", verbose=False
    )

    all_indices = np.arange(len(all_train_for_split))
    
    # Create train-val split. The val split will be further split into val_gallery and val_query splits.
    train_idx, val_idx = train_test_split(
        all_indices, test_size=val_size, random_state=seed, shuffle=True
    )

    train_image_names = [all_train_for_split.annotations[i]["img_name"] for i in train_idx]
    val_image_names = [all_train_for_split.annotations[i]["img_name"] for i in val_idx]

    val_annotations_raw = [deepcopy(all_train_for_split.annotations[i]) for i in val_idx]

    rng = np.random.default_rng(seed)
    pid_to_img_positions = {}
    for position, anno in enumerate(val_annotations_raw):
        valid_pids = set(int(pid) for pid in anno["pids"] if int(pid) != 5555)
        for pid in valid_pids:
            pid_to_img_positions.setdefault(pid, []).append(position)

    val_query_annotations = []
    for pid, positions in sorted(pid_to_img_positions.items()):
        unique_imgs = {val_annotations_raw[p]["img_name"] for p in positions}
        if len(unique_imgs) < 2:
            continue

        positions = list(positions)
        rng.shuffle(positions)
        chosen = positions[0]
        anno = val_annotations_raw[chosen]
        pid_positions = np.where(anno["pids"] == pid)[0]
        if len(pid_positions) == 0:
            continue

        bbox_pos = int(pid_positions[0])
        val_query_annotations.append(
            {
                "img_name": anno["img_name"],
                "img_path": anno["img_path"],
                "boxes": anno["boxes"][bbox_pos : bbox_pos + 1].copy(),
                "pids": np.array([pid], dtype=np.int32),
                "cam_id": anno.get("cam_id", -1),
            }
        )

    if len(val_query_annotations) == 0:
        raise RuntimeError("No valid validation queries found. Try changing split seed or size.")

    train_dataset = build_dataset(
        "PRW",
        dataset_root,
        build_transforms(is_train=True),
        "train",
        image_names=train_image_names,
    )

    val_gallery_dataset = build_dataset(
        "PRW",
        dataset_root,
        build_transforms(is_train=False),
        "val_gallery",
        image_names=val_image_names,
        verbose=True,
    )

    val_query_dataset = build_dataset(
        "PRW",
        dataset_root,
        build_transforms(is_train=False),
        "val_query",
        query_annotations=val_query_annotations,
        verbose=True,
    )

    return train_dataset, val_gallery_dataset, val_query_dataset
