"""SegFormer training loop (Phase 1.5).

Usage:
    python -m src.train --config configs/segformer_b2.yaml --tag segformer_b2
    python -m src.train --config configs/segformer_b2.yaml --tag smoke --smoke

`--smoke` runs 1 epoch on a 100-sample subset to verify the loop end-to-end
before committing to the full 40-epoch run.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.data.dataset import DeepGlobeRoadDataset
from src.data.augmentations import get_train_transforms, get_val_transforms
from src.models.segformer import build_segformer, to_binary_logits
from src.losses import combined_loss

log = logging.getLogger("train")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(log_file: Path):
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(), logging.FileHandler(log_file)):
        h.setFormatter(fmt)
        log.addHandler(h)


def build_cosine_scheduler(optimizer, epochs, steps_per_epoch, warmup_ratio):
    total = epochs * steps_per_epoch
    warmup = max(1, int(total * warmup_ratio))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def validate(model, val_dl, device):
    model.eval()
    inter = union = dice_num = dice_den = 0.0
    for img, mask in val_dl:
        img, mask = img.to(device), mask.to(device)
        with torch.amp.autocast("cuda"):
            logits = to_binary_logits(model(img))
        pred = (torch.sigmoid(logits) > 0.5).float()
        inter += (pred * mask).sum().item()
        union += ((pred + mask) >= 1).float().sum().item()
        dice_num += 2 * (pred * mask).sum().item()
        dice_den += (pred.sum() + mask.sum()).item()
    iou = inter / union if union > 0 else 0.0
    dice = dice_num / dice_den if dice_den > 0 else 0.0
    return {"iou": iou, "dice": dice}


def train_one_epoch(model, train_dl, optimizer, scheduler, scaler, cfg, device, epoch):
    model.train()
    lw = cfg["loss"]
    log_every = cfg["logging"]["log_every"]
    running = 0.0
    for i, (img, mask) in enumerate(train_dl):
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=cfg["optim"]["amp"]):
            logits = to_binary_logits(model(img))
            loss = combined_loss(logits, mask, lw["dice_weight"], lw["bce_weight"],
                                 lw["cldice_weight"], lw["cldice_iters"])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running += loss.item()
        if (i + 1) % log_every == 0:
            lr = scheduler.get_last_lr()[0]
            log.info(f"epoch {epoch} step {i+1}/{len(train_dl)} "
                     f"loss {running/log_every:.4f} lr {lr:.2e}")
            running = 0.0


def save_checkpoint(path, model, optimizer, epoch, metrics, cfg, best_iou):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "metrics": metrics, "cfg": cfg,
                "best_iou": best_iou}, path)


def main(cfg, tag, config_path, smoke=False, resume=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg["logging"]["out_dir"]) / f"{datetime.now():%Y%m%d_%H%M%S}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "train.log")
    shutil.copy(config_path, out_dir / "config.yaml")
    set_seed(42)
    log.info(f"device={device} out_dir={out_dir} smoke={smoke}")

    d = cfg["data"]
    train_ds = DeepGlobeRoadDataset(d["root_dir"], "train",
                                    get_train_transforms(d["image_size"]))
    val_ds = DeepGlobeRoadDataset(d["root_dir"], "val",
                                  get_val_transforms(d["image_size"]))
    if smoke:
        train_ds = Subset(train_ds, range(min(100, len(train_ds))))
        val_ds = Subset(val_ds, range(min(20, len(val_ds))))
        cfg["optim"]["epochs"] = 1

    train_dl = DataLoader(train_ds, batch_size=d["batch_size"], shuffle=True,
                          num_workers=d["num_workers"], pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=d["batch_size"], shuffle=False,
                        num_workers=d["num_workers"], pin_memory=True)
    log.info(f"train={len(train_ds)} val={len(val_ds)} batches/epoch={len(train_dl)}")

    model = build_segformer(cfg["model"]["name"], cfg["model"]["num_labels"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"],
                                  weight_decay=cfg["optim"]["weight_decay"])
    scheduler = build_cosine_scheduler(optimizer, cfg["optim"]["epochs"],
                                       len(train_dl), cfg["optim"]["warmup_ratio"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["optim"]["amp"])

    best_iou = 0.0
    start_epoch = 0
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_iou = ckpt.get("best_iou", ckpt.get("metrics", {}).get("iou", 0.0))
        # fast-forward the LR schedule to where we left off
        for _ in range(start_epoch * len(train_dl)):
            scheduler.step()
        log.info(f"resumed from {resume}: starting epoch {start_epoch}, "
                 f"best_iou={best_iou:.4f}")

    for epoch in range(start_epoch, cfg["optim"]["epochs"]):
        train_one_epoch(model, train_dl, optimizer, scheduler, scaler, cfg, device, epoch)
        metrics = validate(model, val_dl, device)
        log.info(f"== epoch {epoch} val: iou={metrics['iou']:.4f} dice={metrics['dice']:.4f}")
        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            save_checkpoint(out_dir / "best.pth", model, optimizer, epoch, metrics, cfg, best_iou)
            log.info(f"  new best iou={best_iou:.4f} -> best.pth")
        save_checkpoint(out_dir / "last.pth", model, optimizer, epoch, metrics, cfg, best_iou)
    log.info(f"done. best val iou={best_iou:.4f}")
    return best_iou


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch on a 100-sample subset to verify the loop")
    ap.add_argument("--resume", default=None,
                    help="path to last.pth to continue training from")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg, args.tag, args.config, smoke=args.smoke, resume=args.resume)
