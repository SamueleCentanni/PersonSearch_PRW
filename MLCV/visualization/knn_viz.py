from __future__ import annotations

from typing import Tuple

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors


def nearest_neighbors(
    sources: np.ndarray,
    targets: np.ndarray,
    num_neighbors: int,
    algorithm: str = "kd_tree",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute nearest neighbor search from sources to targets."""
    kd_tree = NearestNeighbors(n_neighbors=num_neighbors, algorithm=algorithm, metric="euclidean")
    kd_tree.fit(targets)
    distances, indices = kd_tree.kneighbors(sources)
    return distances, indices


def show_neighbors_prw(
    query_meta: dict,
    gal_meta: list,
    nn_indices: np.ndarray,
    nn_distances: np.ndarray,
    image_root: str,
    topk: int = 3,
) -> None:
    """Show query crop + top-k nearest gallery detections side by side."""
    indices = np.asarray(nn_indices).squeeze()
    distances = np.asarray(nn_distances).squeeze()
    if indices.ndim == 0:
        indices = np.array([int(indices)])
        distances = np.array([float(distances)])

    indices = indices[:topk]
    distances = distances[:topk]

    num_images = len(indices) + 1
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

    # ---- Query crop ----
    query_img = np.array(
        Image.open(os.path.join(image_root, query_meta["img_name"])).convert("RGB")
    )
    qbox = np.asarray(query_meta["box"]).reshape(-1).astype(int)
    x1, y1, x2, y2 = np.clip(qbox, 0, None)
    query_crop = query_img[y1:y2, x1:x2]

    axes[0].imshow(query_crop)
    axes[0].set_title(f"Query\npid={query_meta['pid']}", color="green", fontsize=12)
    axes[0].axis("off")

    # ---- Neighbor crops ----
    for i, (idx_det, dist) in enumerate(zip(indices, distances)):
        g = gal_meta[int(idx_det)]
        gal_img = np.array(
            Image.open(os.path.join(image_root, g["img_name"])).convert("RGB")
        )
        gbox = np.asarray(g["box"]).reshape(-1).astype(int)
        gx1, gy1, gx2, gy2 = np.clip(gbox, 0, None)
        crop = gal_img[gy1:gy2, gx1:gx2]

        axes[i + 1].imshow(crop)
        axes[i + 1].set_title(
            f"#{i+1}  dist={float(dist):.4f}\ndet_score={g['score']:.3f}\n{g['img_name']}",
            fontsize=10,
        )
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.show()


def draw_query_on_image_prw(
    query_meta,
    gallery_meta,
    image_root,
    query_box=None,
    draw_ids: bool = True,
    query_scale: float = 0.25,
    margin: int = 10,
):
    """Draw gallery detection and overlay the query bbox crop at bottom-right."""

    def _xyxy(box):
        arr = np.asarray(box)
        if arr.ndim == 2:
            arr = arr[0]
        arr = arr.reshape(-1)
        if arr.shape[0] != 4:
            raise ValueError(f"Expected 4 values for box, got shape {arr.shape}")
        x1, y1, x2, y2 = map(float, arr)
        return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

    def _resolve_query_box(qmeta, qbox):
        if qbox is not None:
            return _xyxy(qbox)
        if "box" in qmeta and qmeta["box"] is not None:
            return _xyxy(qmeta["box"])
        if "boxes" in qmeta and qmeta["boxes"] is not None:
            return _xyxy(qmeta["boxes"])
        if "gt_box" in qmeta and qmeta["gt_box"] is not None:
            return _xyxy(qmeta["gt_box"])
        raise ValueError(
            "Query bbox missing. Pass query_box=... or add one of ['box','boxes','gt_box'] to query_meta."
        )

    scene_path = os.path.join(image_root, gallery_meta["img_name"])
    scene = np.array(Image.open(scene_path).convert("RGB"))

    gx1, gy1, gx2, gy2 = _xyxy(gallery_meta["box"])
    cv2.rectangle(scene, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)
    gtxt = f"score={float(gallery_meta.get('score', np.nan)):.3f}"
    if draw_ids and ("pid" in gallery_meta):
        gtxt += f" | gpid={gallery_meta['pid']}"
    cv2.putText(scene, gtxt, (gx1, max(15, gy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    query_path = os.path.join(image_root, query_meta["img_name"])
    query_img = np.array(Image.open(query_path).convert("RGB"))
    qx1, qy1, qx2, qy2 = _resolve_query_box(query_meta, query_box)

    hq, wq = query_img.shape[:2]
    qx1, qx2 = np.clip([qx1, qx2], 0, wq)
    qy1, qy2 = np.clip([qy1, qy2], 0, hq)
    if qx2 <= qx1 or qy2 <= qy1:
        raise ValueError("Invalid query bbox after clamping.")

    query_crop = query_img[qy1:qy2, qx1:qx2]

    hs, ws = scene.shape[:2]
    target_w = max(1, int(ws * query_scale))
    qh, qw = query_crop.shape[:2]
    target_h = max(1, int(qh * (target_w / max(1, qw))))

    max_w = max(1, ws - 2 * margin)
    max_h = max(1, hs - 2 * margin)
    scale = min(max_w / max(1, target_w), max_h / max(1, target_h), 1.0)
    new_w = max(1, int(target_w * scale))
    new_h = max(1, int(target_h * scale))
    query_small = cv2.resize(query_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    y1 = max(0, hs - new_h - margin)
    y2 = min(hs, y1 + new_h)
    x1 = max(0, ws - new_w - margin)
    x2 = min(ws, x1 + new_w)

    scene[y1:y2, x1:x2] = query_small[: (y2 - y1), : (x2 - x1)]

    cv2.rectangle(scene, (x1, y1), (x2, y2), (0, 255, 0), 2)
    qtxt = "query-crop"
    if draw_ids and ("pid" in query_meta):
        qtxt += f" | qpid={query_meta['pid']}"
    cv2.putText(scene, qtxt, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    return scene


def show_topk_overlay_prw(
    query_meta,
    gal_meta,
    nn_indices,
    nn_distances,
    image_root,
    topk: int = 3,
    query_box=None,
):
    idxs = np.asarray(nn_indices).squeeze()
    dists = np.asarray(nn_distances).squeeze()
    if idxs.ndim == 0:
        idxs = np.array([int(idxs)])
        dists = np.array([float(dists)])

    idxs = idxs[:topk]
    dists = dists[:topk]

    n = len(idxs)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for i, (ax, idx_det, dist) in enumerate(zip(axes, idxs, dists), start=1):
        g = gal_meta[int(idx_det)]
        canvas = draw_query_on_image_prw(
            query_meta=query_meta,
            gallery_meta=g,
            image_root=image_root,
            query_box=query_box,
            draw_ids=True,
        )
        ax.imshow(canvas)
        ax.axis("off")
        ax.set_title(
            f"#{i} dist={float(dist):.4f}\nimg={g['img_name']}\nscore={float(g.get('score', np.nan)):.3f}",
            color="red",
        )

    plt.tight_layout()
    plt.show()
