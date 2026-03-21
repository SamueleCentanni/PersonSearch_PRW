"""
Deep comparison and visualization utilities for multi-backbone person search evaluation.

Provides:
- Training curve overlays across backbones
- Per-query side-by-side top-k comparison
- Failure / success analysis
- Model agreement heatmap
- Quantitative summary table
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score


# ──────────────────────────────────────────────────────────────
# 1.  Quantitative summary table
# ──────────────────────────────────────────────────────────────

def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def measure_inference_speed(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    num_batches: int = 50,
    warmup: int = 5,
) -> float:
    """
    Measure average inference time (ms) per image.

    Runs `warmup` batches first (not timed), then times `num_batches`.
    """
    model.eval()
    times = []
    for i, (images, targets) in enumerate(dataloader):
        if i >= warmup + num_batches:
            break
        images = [img.to(device) for img in images]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        if i >= warmup:
            times.append((t1 - t0) * 1000 / len(images))  # ms per image
    return float(np.mean(times)) if times else float("nan")


def build_summary_table(
    models: Dict[str, torch.nn.Module],
    results: Dict[str, dict],
    dataloader=None,
    device: torch.device = None,
    measure_speed: bool = True,
) -> pd.DataFrame:
    """
    Build a DataFrame comparing backbones.

    Args:
        models:  {"ResNet50": model_obj, "MobileNetV3Large": model_obj, ...}
        results: {"ResNet50": eval_result_dict, ...}  — from evaluate_performance()
        dataloader: gallery loader for speed measurement (optional).
        device: torch device.
        measure_speed: whether to benchmark inference time.

    Returns:
        pd.DataFrame with columns [Backbone, Total Params, Trainable Params,
                                    mAP (%), Top-1 (%), Inference (ms/img)].
    """
    rows = []
    for name, model in models.items():
        total, trainable = count_parameters(model)
        res = results.get(name, {})
        mAP = float(res.get("mAP", 0)) * 100
        accs = np.asarray(res.get("accs", [0])).ravel()
        top1 = float(accs[0]) * 100 if len(accs) > 0 else 0.0

        speed = float("nan")
        if measure_speed and dataloader is not None and device is not None:
            speed = measure_inference_speed(model, dataloader, device)

        rows.append({
            "Backbone": name,
            "Total Params": f"{total:,}",
            "Trainable Params": f"{trainable:,}",
            "mAP (%)": f"{mAP:.2f}",
            "Top-1 (%)": f"{top1:.2f}",
            "Inference (ms/img)": f"{speed:.1f}" if not np.isnan(speed) else "—",
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# 2.  Training curve overlay
# ──────────────────────────────────────────────────────────────

def compare_training_curves(
    csv_paths: Dict[str, str],
    metrics: Optional[List[str]] = None,
) -> None:
    """
    Overlay training curves from multiple backbone CSVs.

    Args:
        csv_paths: {"ResNet50": "/path/to/csv", ...}
        metrics:   columns to plot (default: train_loss, val_loss, val_mAP, val_top1).
    """
    if metrics is None:
        metrics = ["train_loss", "val_loss", "val_mAP", "val_top1"]

    dfs = {}
    for name, path in csv_paths.items():
        df = pd.read_csv(path)
        # convert metric columns to numeric
        for m in metrics:
            if m in df.columns:
                df[m] = pd.to_numeric(df[m], errors="coerce")
        dfs[name] = df

    loss_metrics = [m for m in metrics if "loss" in m]
    perf_metrics = [m for m in metrics if "loss" not in m]

    n_plots = int(bool(loss_metrics)) + int(bool(perf_metrics))
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]
    ax_idx = 0

    colors = plt.cm.tab10.colors
    styles = ["-", "--", ":"]

    if loss_metrics:
        ax = axes[ax_idx]; ax_idx += 1
        for ci, (name, df) in enumerate(dfs.items()):
            for si, m in enumerate(loss_metrics):
                if m not in df.columns:
                    continue
                subset = df.dropna(subset=[m])
                ax.plot(subset["epoch"], subset[m],
                        label=f"{name} — {m}",
                        color=colors[ci % len(colors)],
                        linestyle=styles[si % len(styles)],
                        marker="o", markersize=3)
        ax.set_title("Loss Curves", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9)

    if perf_metrics:
        ax = axes[ax_idx]
        for ci, (name, df) in enumerate(dfs.items()):
            for si, m in enumerate(perf_metrics):
                if m not in df.columns:
                    continue
                subset = df.dropna(subset=[m])
                vals = subset[m]
                # scale to % if values are in [0,1]
                if vals.max() <= 1.0:
                    vals = vals * 100
                ax.plot(subset["epoch"], vals,
                        label=f"{name} — {m}",
                        color=colors[ci % len(colors)],
                        linestyle=styles[si % len(styles)],
                        marker="^", markersize=3)
        ax.set_title("Validation Metrics", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Percentage (%)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────
# 3.  Per-query side-by-side top-k comparison
# ──────────────────────────────────────────────────────────────

def _crop_from_image(image_root: str, img_name: str, box) -> np.ndarray:
    """Load image and crop the given box region."""
    img = np.array(Image.open(os.path.join(image_root, img_name)).convert("RGB"))
    x1, y1, x2, y2 = [int(round(v)) for v in np.asarray(box).ravel()[:4]]

    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    return img[y1:y2, x1:x2]


def compare_topk_per_query(
    all_results: Dict[str, dict],
    query_idx: int,
    image_root: str,
    topk: int = 5,
) -> None:
    """
    Side-by-side top-k gallery matches for the same query across models.

    - Query is shown as a cropped image
    - Gallery shows full image + detected person bounding box
    - Detected person is zoomed using a Matplotlib inset with connecting lines
      (Matplotlib documentation-style, CORRECT orientation)
    """

    model_names = list(all_results.keys())
    n_models = len(model_names)

    # ------------------------------------------------------------
    # Query info (shared across models)
    # ------------------------------------------------------------
    first_res = all_results[model_names[0]]["results"][query_idx]
    query_img_name = first_res["query_img"]
    query_roi = first_res["query_roi"]
    query_pid = first_res.get("query_pid", "N/A")

    query_crop = _crop_from_image(image_root, query_img_name, query_roi)

    n_cols = topk + 1
    fig, axes = plt.subplots(
        n_models,
        n_cols,
        figsize=(5 * n_cols, 4.5 * n_models),
    )

    if n_models == 1:
        axes = [axes]

    # ============================================================
    # Main loop
    # ============================================================
    for row, model_name in enumerate(model_names):
        res = all_results[model_name]["results"][query_idx]
        gallery = res["gallery"][:topk]

        # ------------------ Query column ------------------
        axes[row][0].imshow(query_crop, origin="upper")
        axes[row][0].set_title(
            f"Query PID: {query_pid}\n({model_name})",
            fontsize=11,
            fontweight="bold",
        )
        axes[row][0].axis("off")

        # ------------------ Gallery columns ------------------
        for col, g in enumerate(gallery, start=1):
            ax = axes[row][col]

            full_img = np.array(
                Image.open(os.path.join(image_root, g["img"])).convert("RGB")
            )
            h, w = full_img.shape[:2]

            ax.imshow(full_img, origin="upper")

            # ROI predicted by SeqNet
            x1, y1, x2, y2 = [int(round(v)) for v in np.asarray(g["roi"]).ravel()[:4]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            correct = bool(g.get("correct", 0))
            color = "lime" if correct else "red"
            gallery_pid = g.get("pid", "N/A")

            # ------------------ Bounding box ------------------
            rect_outer = mpatches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=color,
                linewidth=2.5,
            )
            rect_inner = mpatches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="white",
                linewidth=0.8,
            )
            ax.add_patch(rect_outer)
            ax.add_patch(rect_inner)

            # ------------------ Inset zoom (CORRECT) ------------------
            axins = ax.inset_axes(
                [0.55, 0.05, 0.4, 0.4],
                xlim=(x1, x2),
                ylim=(y2, y1),   # ✅ CRUCIAL FIX
                xticklabels=[],
                yticklabels=[],
            )

            axins.imshow(full_img, origin="upper")

            for spine in axins.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(3)

            ax.indicate_inset_zoom(
                axins,
                edgecolor=color,
                linewidth=2,
                alpha=0.85,
            )

            label = "CORRECT" if correct else "WRONG"
            ax.set_title(
                f"#{col} {label}\nPID: {gallery_pid} | Score: {g['score']:.3f}",
                fontsize=10,
                fontweight="bold",
                color="green" if correct else "red",
            )
            ax.axis("off")

    # ------------------ Legend & layout ------------------
    green_patch = mpatches.Patch(color="green", label="Correct match")
    red_patch = mpatches.Patch(color="red", label="Wrong match")

    fig.legend(
        handles=[green_patch, red_patch],
        loc="lower center",
        ncol=2,
        fontsize=11,
    )

    plt.suptitle(
        f"Query #{query_idx}: {query_img_name}",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout()
    plt.show()

# ──────────────────────────────────────────────────────────────
# 4.  Failure / success analysis
# ──────────────────────────────────────────────────────────────

def per_query_correctness(results: dict) -> np.ndarray:
    """
    Return boolean array: True if top-1 match is correct for each query.
    """
    return np.array([
        bool(r["gallery"][0]["correct"]) if r["gallery"] else False
        for r in results["results"]
    ])


def find_disagreements(
    all_results: Dict[str, dict],
) -> Dict[str, List[int]]:
    """
    Find query indices where models disagree on top-1 correctness.

    Returns dict with keys:
        "all_correct":  queries where every model got top-1 right
        "all_wrong":    queries where every model got top-1 wrong
        "disagree":     queries where models disagree
    Also per-model keys like "ResNet50_only_correct" etc.
    """
    model_names = list(all_results.keys())
    n_queries = len(all_results[model_names[0]]["results"])

    correct_per_model = {
        name: per_query_correctness(all_results[name])
        for name in model_names
    }

    stacked = np.stack(list(correct_per_model.values()), axis=0)  # (n_models, n_queries)
    all_correct_mask = stacked.all(axis=0)
    all_wrong_mask = (~stacked).all(axis=0)
    disagree_mask = ~all_correct_mask & ~all_wrong_mask

    out = {
        "all_correct": np.where(all_correct_mask)[0].tolist(),
        "all_wrong": np.where(all_wrong_mask)[0].tolist(),
        "disagree": np.where(disagree_mask)[0].tolist(),
    }

    # per-model unique successes
    for i, name in enumerate(model_names):
        only_this = stacked[i] & (~np.delete(stacked, i, axis=0).any(axis=0))
        out[f"{name}_unique_correct"] = np.where(only_this)[0].tolist()

    return out


def print_disagreement_summary(
    all_results: Dict[str, dict],
) -> Dict[str, List[int]]:
    """Print and return disagreement analysis."""
    info = find_disagreements(all_results)
    n_queries = len(list(all_results.values())[0]["results"])

    print(f"Total queries: {n_queries}")
    print(f"  All models correct (top-1): {len(info['all_correct'])} ({100*len(info['all_correct'])/n_queries:.1f}%)")
    print(f"  All models wrong   (top-1): {len(info['all_wrong'])} ({100*len(info['all_wrong'])/n_queries:.1f}%)")
    print(f"  Models disagree    (top-1): {len(info['disagree'])} ({100*len(info['disagree'])/n_queries:.1f}%)")

    for key in info:
        if key.endswith("_unique_correct"):
            model_name = key.replace("_unique_correct", "")
            print(f"  {model_name} uniquely correct: {len(info[key])}")

    return info


def show_failure_cases(
    all_results: Dict[str, dict],
    image_root: str,
    category: str = "all_wrong",
    max_show: int = 5,
    topk: int = 3,
) -> None:
    """
    Visualize queries from a specific category (all_wrong, disagree, etc.).

    Args:
        category: one of "all_wrong", "disagree", "all_correct", or
                  "{ModelName}_unique_correct"
        max_show: max number of queries to display.
    """
    info = find_disagreements(all_results)
    indices = info.get(category, [])

    if not indices:
        print(f"No queries found in category '{category}'.")
        return

    print(f"Showing up to {max_show} queries from '{category}' (total: {len(indices)})")
    for idx in indices[:max_show]:
        compare_topk_per_query(all_results, idx, image_root, topk=topk)


# ──────────────────────────────────────────────────────────────
# 5.  Model agreement heatmap
# ──────────────────────────────────────────────────────────────

def plot_agreement_matrix(
    all_results: Dict[str, dict],
) -> None:
    """
    Heatmap showing pairwise top-1 agreement rate between models.
    """
    model_names = list(all_results.keys())
    n = len(model_names)

    correct = {
        name: per_query_correctness(all_results[name])
        for name in model_names
    }

    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            agree = (correct[model_names[i]] == correct[model_names[j]]).mean()
            matrix[i, j] = agree * 100

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="YlGn", vmin=50, vmax=100)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_yticklabels(model_names, fontsize=10)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i, j]:.1f}%",
                    ha="center", va="center", fontsize=12, fontweight="bold")

    plt.colorbar(im, label="Agreement (%)")
    ax.set_title("Top-1 Agreement Between Models", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────
# 6.  Per-query AP distribution
# ──────────────────────────────────────────────────────────────

def plot_ap_distribution(
    all_results: Dict[str, dict],
) -> None:
    """
    Overlay histogram of per-query AP for each model.
    Useful to see the spread — are failures concentrated or spread out?
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, res in all_results.items():
        # reconstruct per-query AP from results
        aps = []
        for r in res["results"]:
            gallery = r["gallery"]
            y_true = [g["correct"] for g in gallery]
            y_score = [g["score"] for g in gallery]
            if sum(y_true) > 0:
                aps.append(average_precision_score(y_true, y_score))
            else:
                aps.append(0.0)
        ax.hist(aps, bins=20, alpha=0.5, label=f"{name} (mean={np.mean(aps):.3f})", edgecolor="black")

    ax.set_xlabel("Average Precision", fontsize=12)
    ax.set_ylabel("Number of Queries", fontsize=12)
    ax.set_title("Per-Query AP Distribution", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────
# 7.  Score distribution: correct vs. wrong
# ──────────────────────────────────────────────────────────────

def plot_score_distributions(
    all_results: Dict[str, dict],
) -> None:
    """
    For each model, show the distribution of similarity scores
    for correct vs. wrong top-1 matches.
    """
    model_names = list(all_results.keys())
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, model_names):
        correct_scores = []
        wrong_scores = []
        for r in all_results[name]["results"]:
            if not r["gallery"]:
                continue
            top1 = r["gallery"][0]
            if top1["correct"]:
                correct_scores.append(top1["score"])
            else:
                wrong_scores.append(top1["score"])

        ax.hist(correct_scores, bins=25, alpha=0.6, label="Correct", color="green", edgecolor="black")
        ax.hist(wrong_scores, bins=25, alpha=0.6, label="Wrong", color="red", edgecolor="black")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Similarity Score")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Count")
    plt.suptitle("Top-1 Score Distribution: Correct vs Wrong", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
