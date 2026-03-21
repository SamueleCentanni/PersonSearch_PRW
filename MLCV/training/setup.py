from __future__ import annotations

import torch

from dataset.build import build_train_loader, build_test_loader, custom_collate_fn, search_num_workers
from dataset.splits import build_prw_train_val_split


def build_prw_loaders(cfg, dataset_root: str, val_size: int = 570, seed: int = 0, num_workers: int | None = None):
    """Build DataLoaders for the PRW dataset, including train, validation, and test loaders."""
    
    # train and validation datasets
    train_dataset, val_gallery_dataset, val_query_dataset = build_prw_train_val_split(
        dataset_root, val_size=val_size, seed=seed
    )

    if num_workers is None:
        print("\n================================")
        print("Searching for optimal number of workers...\n")
        num_workers = search_num_workers(dataset=train_dataset, batch_size=cfg.INPUT.BATCH_SIZE_TRAIN)
        print("================================\n")

    train_loader = build_train_loader(cfg=cfg, dataset=train_dataset, num_workers=num_workers)
    
    # I create the val_loader to compute the validation loss
    val_loader = build_train_loader(cfg=cfg, dataset=val_gallery_dataset, split="validation", num_workers=num_workers)


    # I compute the val_gallery_loader and val_query_loader to compute the validation mAP and top1.
    # The val_gallery_loader and val_query_loader are used to emulate the test setting, where we have a gallery set and a query set.
    val_gallery_loader = torch.utils.data.DataLoader(
        val_gallery_dataset,
        batch_size=cfg.INPUT.BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=cfg.INPUT.NUM_WORKERS_TEST,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    val_query_loader = torch.utils.data.DataLoader(
        val_query_dataset,
        batch_size=cfg.INPUT.BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=cfg.INPUT.NUM_WORKERS_TEST,
        pin_memory=True,
        collate_fn=custom_collate_fn,
    )

    test_gallery_loader, test_query_loader = build_test_loader(cfg=cfg, dataset_root=dataset_root)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "val_gallery_loader": val_gallery_loader,
        "val_query_loader": val_query_loader,
        "test_gallery_loader": test_gallery_loader,
        "test_query_loader": test_query_loader,
    }
