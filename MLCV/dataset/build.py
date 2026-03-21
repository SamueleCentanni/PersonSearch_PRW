from __future__ import annotations

import os
import time
from typing import Iterable

import numpy as np
import torch
from tabulate import tabulate

from dataset.prw import PRW
from dataset.transforms import build_transforms


def custom_collate_fn(batch):
    """Variable-size images can't be stacked — return as tuple of lists."""
    return tuple(zip(*batch))


def search_num_workers(dataset, batch_size: int, limit_batches: int = 50) -> int:
    """Search for an efficient num_workers by timing a limited number of batches."""
    max_workers = os.cpu_count() or 0
    candidates = list(range(max_workers, -1, -1))

    best_workers = 0
    best_time = float("inf")

    print(f"Available CPU Cores: {max_workers}")
    print(f"Testing candidates: {candidates} (Stop after {limit_batches} batches)")

    for num_workers in candidates:
        print(f"\n--- Testing num_workers={num_workers} ---")

        try:
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                collate_fn=custom_collate_fn,
                pin_memory=True,
            )

            start_event = time.time()
            for i, _ in enumerate(loader):
                if i >= limit_batches:
                    break

            total_time = time.time() - start_event
            avg_time = total_time / limit_batches
            print(f"Success! Total time: {total_time:.2f}s | Avg/batch: {avg_time:.4f}s")

            if total_time < best_time:
                best_time = total_time
                best_workers = num_workers

        except RuntimeError as e:
            print(f"MEMORY CRASH with workers={num_workers}. Error: {e}")
            print("This setting consumes too much RAM for Colab.")
        except Exception as e:
            print(f"Generic error: {e}")

    print(f"\nWinner: num_workers = {best_workers}")
    return best_workers


def create_small_table(small_dict):
    """Create a small table using the keys of small_dict as headers."""
    keys, values = tuple(zip(*small_dict.items()))
    return tabulate(
        [values],
        headers=keys,
        tablefmt="pipe",
        floatfmt=".3f",
        stralign="center",
        numalign="center",
    )


def print_statistics(dataset) -> None:
    """Print dataset statistics."""
    num_imgs = len(dataset.annotations)
    num_boxes = 0
    pid_set = set()
    for anno in dataset.annotations:
        num_boxes += anno["boxes"].shape[0]
        for pid in anno["pids"]:
            pid_set.add(pid)

    statistics = {
        "dataset": dataset.name,
        "split": dataset.split,
        "num_images": num_imgs,
        "num_boxes": num_boxes,
    }

    if dataset.split in ("query", "val_query"):
        pid_list = sorted(list(pid_set))
        if pid_list:
            statistics.update(
                {
                    "num_labeled_pids": len(pid_list),
                    "min_labeled_pid": int(min(pid_list)),
                    "max_labeled_pid": int(max(pid_list)),
                }
            )
    else:
        pid_list = sorted(list(pid_set))
        if pid_list:
            unlabeled_pid = pid_list[-1]
            pid_list = pid_list[:-1]
            if pid_list:
                statistics.update(
                    {
                        "num_labeled_pids": len(pid_list),
                        "min_labeled_pid": int(min(pid_list)),
                        "max_labeled_pid": int(max(pid_list)),
                        "unlabeled_pid": int(unlabeled_pid),
                    }
                )

    print(f"=> {dataset.name}-{dataset.split} loaded:\n" + create_small_table(statistics))


def build_dataset(dataset_name, root, transforms, split, verbose: bool = True, **kwargs):
    """Factory function to build a dataset."""
    if dataset_name == "PRW":
        dataset = PRW(root, transforms, split, **kwargs)
    else:
        raise NotImplementedError(f"Unknow dataset: {dataset_name}")

    if verbose:
        print_statistics(dataset)
    return dataset


def build_train_loader(cfg, dataset, split: str = "train", num_workers: int | None = None):
    """Build a DataLoader for training or validation."""
    if num_workers is None:
        num_workers = search_num_workers(dataset=dataset, batch_size=cfg.INPUT.BATCH_SIZE_TRAIN)

    if split == "validation":
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.INPUT.BATCH_SIZE_TRAIN,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=custom_collate_fn,
        )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.INPUT.BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn,
    )


def build_test_loader(cfg, dataset_root: str):
    """Build DataLoaders for the gallery and query sets."""
    transforms = build_transforms(is_train=False)
    gallery_set = build_dataset("PRW", dataset_root, transforms, "gallery")
    query_set = build_dataset("PRW", dataset_root, transforms, "query")

    gallery_loader = torch.utils.data.DataLoader(
        gallery_set,
        batch_size=cfg.INPUT.BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=cfg.INPUT.NUM_WORKERS_TEST,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )
    query_loader = torch.utils.data.DataLoader(
        query_set,
        batch_size=cfg.INPUT.BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=cfg.INPUT.NUM_WORKERS_TEST,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )
    return gallery_loader, query_loader
