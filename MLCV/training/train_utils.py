import gc
import math
import sys
import torch
from torch import autocast
from tqdm import tqdm
import wandb


def clean_gpu():
    """
    forces garbage collection and clears gpu cache.
    """
    gc.collect()
    torch.cuda.empty_cache()
    # optional: reset peak stats
    torch.cuda.reset_peak_memory_stats()


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """Official SeqNet warmup: linearly ramp LR from base_lr*warmup_factor to base_lr."""
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def to_device(images, targets, device):
    images = [img.to(device) for img in images]
    for t in targets:
        t["boxes"] = t["boxes"].to(device)
        t["labels"] = t["labels"].to(device)
    return images, targets


def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler, cfg, use_amp=False):
    """
    Trains the model for one epoch.
    Includes gradient clipping. If warmup_scheduler is provided, steps it each iteration.
    """
    model.train()
    loss_sum = 0.0
    clip_value = cfg.SOLVER.CLIP_GRADIENTS

    warmup_scheduler = None
    if epoch == 0:
        warmup_factor = cfg.SOLVER.WARMUP_FACTOR # 1.0 / 1000
        warmup_iters = max(1, len(data_loader) - 1)
        warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    pbar = tqdm(data_loader, desc=f"train epoch {epoch}", leave=False)

    for step, (images, targets) in enumerate(pbar):
        images, targets = to_device(images, targets, device)

        with autocast(enabled=use_amp):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        scaler.scale(losses).backward()

        # Gradient clipping (before optimizer step)
        if clip_value > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        scaler.step(optimizer)
        scaler.update()

        if epoch == 0:
            warmup_scheduler.step()  # use scheduler only for the first epoch (warmup phase)

        # W&B step logging
        wandb.log({
            "train/total_loss": loss_value,
            "train/reid_loss": loss_dict.get("loss_box_reid", torch.tensor(0)).item() if isinstance(loss_dict.get("loss_box_reid", 0), torch.Tensor) else 0,
            "train/rpn_cls": loss_dict.get("loss_rpn_cls", torch.tensor(0)).item() if isinstance(loss_dict.get("loss_rpn_cls", 0), torch.Tensor) else 0,
            "train/rpn_reg": loss_dict.get("loss_rpn_reg", torch.tensor(0)).item() if isinstance(loss_dict.get("loss_rpn_reg", 0), torch.Tensor) else 0,
            "train/box_cls": loss_dict.get("loss_box_cls", torch.tensor(0)).item() if isinstance(loss_dict.get("loss_box_cls", 0), torch.Tensor) else 0,
            "train/box_reg": loss_dict.get("loss_box_reg", torch.tensor(0)).item() if isinstance(loss_dict.get("loss_box_reg", 0), torch.Tensor) else 0,
            "train/lr": optimizer.param_groups[0]["lr"],
            "epoch": epoch
        })

        loss_sum += loss_value
        pbar.set_postfix({"loss": f"{loss_value:.3f}", "lr": f"{optimizer.param_groups[0]['lr']:.6f}"})

    return loss_sum / len(data_loader)


def validate_one_epoch(model, data_loader, device, epoch, use_amp=True):
    """
    Computes validation loss (model kept in train mode so it returns loss dict, not detections).
    """
    model.train()
    loss_sum = 0.0

    pbar = tqdm(data_loader, desc=f"val epoch {epoch}", leave=False)

    with torch.no_grad():
        for images, targets in pbar:
            images, targets = to_device(images, targets, device)

            with autocast(enabled=use_amp):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            loss_value = losses.item()
            loss_sum += loss_value
            pbar.set_postfix({"val_loss": f"{loss_value:.3f}"})

            wandb.log({
                "val/total_loss": loss_value,
                "epoch": epoch
            })

    return loss_sum / len(data_loader)