import numpy as np
import torch
from tqdm import tqdm
import wandb

from copy import deepcopy

from testing.eval_search_prw import eval_search_prw
from training.train_utils import to_device


@torch.no_grad()
def evaluate_performance(
    model, gallery_loader, query_loader, device, use_gt=False, use_cbgm=False
):
    """
    Args:
        use_gt (bool, optional): Whether to use GT as detection results to verify the upper
                                bound of person search performance. Defaults to False.
        use_cbgm (bool, optional): Whether to use Context Bipartite Graph Matching algorithm.
                                Defaults to False.
    """
    model.eval()

    print("[Eval 1/3] Extracting gallery detections + embeddings...")
    gallery_dets, gallery_feats = [], []
    for images, targets in tqdm(
        gallery_loader,
        total=len(gallery_loader),
        desc="Eval 1/3 | gallery",
        leave=True,
        dynamic_ncols=True,
    ):
        images, targets = to_device(images, targets, device)
        if not use_gt:
            outputs = model(images)
        else:
            boxes = targets[0]["boxes"] # use GT boxes as detection results
            n_boxes = boxes.size(0)
            embeddings = model(images, targets)
            outputs = [
                {
                    "boxes": boxes,
                    "embeddings": torch.cat(embeddings),
                    "labels": torch.ones(n_boxes).to(device),
                    "scores": torch.ones(n_boxes).to(device),
                }
            ]

        for output in outputs:
            box_w_scores = torch.cat([output["boxes"], output["scores"].unsqueeze(1)], dim=1)
            gallery_dets.append(box_w_scores.cpu().numpy())
            gallery_feats.append(output["embeddings"].cpu().numpy())

    # regarding query image as gallery to detect all people
    # i.e. query person + surrounding people (context information)
    # For now, let's comment this part since I don't use CBGM -> when I will,
    # add if else statement

    query_dets, query_feats = [], []

    if use_cbgm:
        for images, targets in tqdm(query_loader, ncols=0):
            images, targets = to_device(images, targets, device)
            # targets will be modified in the model, so deepcopy it
            outputs = model(images, deepcopy(targets), query_img_as_gallery=True)

            # consistency check
            gt_box = targets[0]["boxes"].squeeze()
            assert (
                gt_box - outputs[0]["boxes"][0]
            ).sum() <= 0.001, "GT box must be the first one in the detected boxes of query image"

            for output in outputs:
                box_w_scores = torch.cat([output["boxes"], output["scores"].unsqueeze(1)], dim=1)
                query_dets.append(box_w_scores.cpu().numpy())
                query_feats.append(output["embeddings"].cpu().numpy())

    # extract the features of query boxes (the queries images are already cropped in the dataset)
    print("[Eval 2/3] Extracting query embeddings...")
    query_box_feats = []
    for images, targets in tqdm(
        query_loader,
        total=len(query_loader),
        desc="Eval 2/3 | query",
        leave=True,
        dynamic_ncols=True,
    ):
        images, targets = to_device(images, targets, device)
        embeddings = model(images, targets)
        assert len(embeddings) == 1, "batch size in test phase should be 1"
        query_box_feats.append(embeddings[0].cpu().numpy())

    return eval_search_prw(
        gallery_loader.dataset,
        query_loader.dataset,
        gallery_dets,
        gallery_feats,
        query_box_feats,
        query_dets,
        query_feats,
        cbgm=use_cbgm,
    )