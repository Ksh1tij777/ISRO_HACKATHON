"""Visualize model predictions as PNGs (CV sanity check / pitch visuals).

Runs the trained model on validation tiles and saves a grid:
    input photo | ground truth | prediction | overlay (pred=red, GT=green)

Usage:
    python scripts/visualize_predictions.py \
        --checkpoint runs/<run>/best.pth \
        --root data/deepglobe/train/archive/train \
        --n 6 --out results/predictions_val.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.inference import load_model, normalize
from src.data.dataset import DeepGlobeRoadDataset


def load_pair(sat_path, mask_path, size=512):
    img = cv2.cvtColor(cv2.imread(sat_path), cv2.COLOR_BGR2RGB)
    mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    img = cv2.resize(img, (size, size))
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    return img, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--root", default="data/deepglobe/train/archive/train")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="results/predictions_val.png")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device).eval()

    ds = DeepGlobeRoadDataset(args.root, split="val")
    # pick samples that actually contain road, for a meaningful picture
    picks = []
    for sat, mask in ds.pairs:
        _, m = load_pair(sat, mask)
        if m.mean() > 0.03:
            picks.append((sat, mask))
        if len(picks) >= args.n:
            break

    fig, axes = plt.subplots(len(picks), 4, figsize=(14, 3.4 * len(picks)))
    if len(picks) == 1:
        axes = axes[None, :]
    cols = ["input", "ground truth", "prediction", "overlay (pred=red, GT=green)"]

    for r, (sat, mask) in enumerate(picks):
        img, gt = load_pair(sat, mask)
        x = torch.from_numpy(normalize(img.transpose(2, 0, 1))).unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device == "cuda")):
            prob = torch.softmax(model(x), dim=1)[0, 1]
        pred = (prob.float().cpu().numpy() > args.threshold).astype(np.uint8)

        overlay = img.copy()
        overlay[gt == 1] = [0, 255, 0]
        overlay[pred == 1] = [255, 0, 0]  # pred drawn on top

        for c, im in enumerate([img, gt * 255, pred * 255, overlay]):
            cmap = "gray" if c in (1, 2) else None
            axes[r, c].imshow(im, cmap=cmap)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(cols[c], fontsize=11)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}  ({len(picks)} samples)")


if __name__ == "__main__":
    main()
