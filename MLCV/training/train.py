import csv
import os
import traceback
import numpy as np
import torch
import torch.optim as optim
import wandb

from model.seqnet import SeqNet
from training.train_utils import train_one_epoch, validate_one_epoch, clean_gpu
from utils.seed import fix_random
from testing.evaluation import evaluate_performance
import csv


def run_experiment(cfg, device, loaders, use_amp=False, backbone_name="resnet50"):
    """
    Runs a full training experiment with the given configuration and data loaders.
    Logs metrics to W&B and saves checkpoints.
    """
    # Pretty name for logs and W&B (e.g. "MobileNetV3Large", "ConvNeXtSmall")
    _BACKBONE_NAME = {
        "resnet50": "ResNet50",
        "mobilenet_v3_large": "MobileNetV3Large",
        "convnext_small": "ConvNeXtSmall",
    }
    backbone = _BACKBONE_NAME.get(backbone_name, backbone_name)

    working_dir = "/content/drive/MyDrive/Assignment_MLCV"
    exp_name = f"SeqNet_{backbone}_PRW"
    eval_period = cfg.SOLVER.EVAL_PERIOD
    ckpt_period = cfg.SOLVER.CKPT_PERIOD
    max_epochs = cfg.SOLVER.MAX_EPOCHS

    fix_random(seed=0)

    if wandb.run is not None:
        wandb.finish()

    run = wandb.init(
        project="PRW-PersonSearch",
        name=exp_name,
        config={
            "backbone": backbone_name,
            "backbone_pretty": backbone,
            "epochs": max_epochs,
            "base_lr": cfg.SOLVER.BASE_LR,
            "lr_milestones": cfg.SOLVER.LR_DECAY_MILESTONES,
            "gamma": cfg.SOLVER.GAMMA,
            "warmup_factor": cfg.SOLVER.WARMUP_FACTOR,
            "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
            "clip_gradients": cfg.SOLVER.CLIP_GRADIENTS,
            "batch_size": cfg.INPUT.BATCH_SIZE_TRAIN,
            "min_size": cfg.INPUT.MIN_SIZE,
            "max_size": cfg.INPUT.MAX_SIZE,
            "lut_size": cfg.MODEL.LOSS.LUT_SIZE,
            "cq_size": cfg.MODEL.LOSS.CQ_SIZE,
            "mixed_precision": use_amp,
            "eval_period": eval_period,
            "ckpt_period": ckpt_period,
            "selection_metric": "val_mAP",
        },
        reinit=True,
    )

    # Optimizer per backbone (matching original paper recipes):
    #   ResNet     → SGD (SeqNet default)
    #   MobileNet  → RMSprop with 0.9 momentum (MobileNetV3 paper)
    #   ConvNeXt   → AdamW with weight_decay 1e-8 (ConvNeXt paper)
    scheduler_name = "MultiStepLR"
    if backbone_name == "convnext_small":
        optimizer_name = "AdamW"
        scheduler_name = "CosineAnnealingLR"
    else:
        optimizer_name = "SGD"

    # Effective hyperparameters (ConvNeXt has its own tuned recipe).
    effective_lr = cfg.SOLVER.BASE_LR
    effective_wd = cfg.SOLVER.WEIGHT_DECAY
    if backbone_name == "convnext_small":
        effective_lr = cfg.SOLVER.CONVNEXT_BASE_LR
        effective_wd = cfg.SOLVER.CONVNEXT_WEIGHT_DECAY

    print(f"\n{'='*60}")
    print(f"  Experiment : {exp_name}")
    print(f"  Backbone   : {backbone} ({backbone_name})")
    print(f"  Optimizer  : {optimizer_name}")
    print(f"  Scheduler  : {scheduler_name}")
    print(f"  Epochs     : {max_epochs}")
    if backbone_name == "convnext_small":
        print(f"  LR         : {effective_lr}")
        print(f"  WD         : {effective_wd}")
    else:
        print(f"  LR         : {effective_lr} → decay at {cfg.SOLVER.LR_DECAY_MILESTONES} (γ={cfg.SOLVER.GAMMA})")
        print(f"  WD         : {effective_wd}")
    print(f"  Batch size : {cfg.INPUT.BATCH_SIZE_TRAIN}")
    print(f"  Val eval   : every {eval_period} epoch(s)")
    print(f"  Checkpoint : every {ckpt_period} epoch(s)")
    print(f"  AMP        : {use_amp}")
    print(f"{'='*60}")

    try:
        model = SeqNet(cfg, backbone_name=backbone_name)
        model.to(device)

        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters : {num_params:,} total | {num_trainable:,} trainable")
        print(f"{'='*60}\n")

        params = [p for p in model.parameters() if p.requires_grad]

        if backbone_name == "convnext_small":
            optimizer = optim.AdamW(
                params,
                lr=effective_lr,
                weight_decay=effective_wd,
            )
        else:
            optimizer = optim.SGD(
                params,
                lr=effective_lr,
                momentum=cfg.SOLVER.SGD_MOMENTUM,
                weight_decay=effective_wd,
            )

        if backbone_name == "convnext_small":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epochs,
                eta_min=effective_lr * 0.1,
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=cfg.SOLVER.LR_DECAY_MILESTONES,
                gamma=cfg.SOLVER.GAMMA,
            )

        save_dir = os.path.join(working_dir, "checkpoints")
        sub_save_dir = os.path.join(save_dir, exp_name)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(sub_save_dir, exist_ok=True)

        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        best_val_mAP = float("-inf")
        best_epoch = -1
        best_model_path = os.path.join(sub_save_dir, f"{exp_name}_best_val_map.pth")

        # CSV logging
        csv_path = os.path.join(sub_save_dir, f"{exp_name}_metrics.csv")
        csv_fields = ["epoch", "train_loss", "val_loss", "lr", "val_mAP", "val_top1"]
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        csv_writer.writeheader()
        print(f"  CSV log    : {csv_path}")
        
        # define loaders from the input dictionary
        train_loader = loaders["train_loader"]
        val_loader = loaders["val_loader"]
        val_gallery_loader = loaders["val_gallery_loader"]
        val_query_loader = loaders["val_query_loader"]

        for epoch in range(0, max_epochs):
            train_loss = train_one_epoch(
                model, optimizer, train_loader, device, epoch, scaler, cfg, use_amp
            )

            val_loss = validate_one_epoch(
                model, val_loader, device, epoch, use_amp=use_amp
            )

            lr_scheduler.step()

            log_dict = {
                "train/epoch_loss": train_loss,
                "val/epoch_loss": val_loss,
                "train/lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
            }

            print("")
            print(
                f"[{backbone}] Epoch {epoch+1}/{max_epochs} — "
                f"Train: {train_loss:.4f} | ValLoss: {val_loss:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
            print("-----")

            # validation retrieval protocol (val_gallery_loader / val_query_loader)
            if (epoch + 1) % eval_period == 0 or epoch == max_epochs - 1:
                print(f">>> Epoch {epoch+1}: Validation -> Running person-search evaluation...")
                eval_results = evaluate_performance(
                    model, val_gallery_loader, val_query_loader, device,
                )
                val_mAP = float(eval_results["mAP"])
                val_top1 = float(np.asarray(eval_results["accs"]).ravel()[0])

                log_dict["val_retrieval/mAP"] = val_mAP
                log_dict["val_retrieval/top1"] = val_top1
                print(f">>> Epoch {epoch}: val mAP={val_mAP:.2%} | val top-1={val_top1:.2%}")

                if val_mAP > best_val_mAP:
                    best_val_mAP = val_mAP
                    best_epoch = epoch
                    torch.save(model.state_dict(), best_model_path)
                    print(f">>> New best val mAP! Checkpoint saved to {best_model_path}")
                model.train()

            # Write CSV row
            csv_row = {
                "epoch": epoch + 1,
                "train_loss": f"{train_loss:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                "val_mAP": f"{log_dict.get('val_retrieval/mAP', '')}",
                "val_top1": f"{log_dict.get('val_retrieval/top1', '')}",
            }
            csv_writer.writerow(csv_row)
            csv_file.flush()

            wandb.log(log_dict)

            if (epoch + 1) % ckpt_period == 0 or epoch == max_epochs - 1:
                ckpt_path = os.path.join(sub_save_dir, f"epoch_{epoch}.pth")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                    },
                    ckpt_path,
                )
                print(f">>> Checkpoint saved to {ckpt_path}")
                print("===================")

        print(f"\n[{backbone}] Training complete. Best validation mAP: {best_val_mAP:.2%} (epoch {best_epoch+1})")

        try:
            artifact = wandb.Artifact(name=f"model-{exp_name}", type="model")
            artifact.add_file(best_model_path)
            run.log_artifact(artifact)
            artifact.wait()
            print("Best model uploaded to W&B artifacts.")
        except Exception as e:
            print(f"W&B artifact upload failed: {e}")

    except Exception as e:
        print(f"CRASH in experiment: {e}")
        traceback.print_exc()
        return float("inf")

    finally:
        try:
            csv_file.close()
        except Exception:
            pass
        wandb.finish()
        try:
            del model, optimizer, scaler
        except Exception:
            pass
        clean_gpu()
        print("GPU memory cleared.")
    
    return best_val_mAP