"""Tiled inference on the Jaipur Sentinel-2 image (Phase 1.7).

Produces the integration-contract file:
    data/inference/jaipur/predicted_mask.tif
    - single-channel uint8 GeoTIFF, values {0, 1}
    - same CRS + transform as the input
    - compress="lzw"

Handles the real-data quirks (see memory/jaipur-data-facts):
- float64 Sentinel-2 reflectance -> per-band 2/98-percentile stretch to uint8,
  so the input distribution roughly matches the DeepGlobe RGB the model trained on
- NaN nodata pixels -> filled for inference, then forced to background (0) in output
- arbitrary CRS/transform preserved (Jaipur is EPSG:4326, not 3857)

Usage:
    python -m src.inference \
        --checkpoint runs/<run>/best.pth \
        --input data/inference/jaipur/jaipur_s2_rgb.tif \
        --output data/inference/jaipur/predicted_mask.tif
"""
from __future__ import annotations

import argparse

import numpy as np
import rasterio
import torch

from src.models.segformer import build_segformer

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def stretch_to_uint8(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(3,H,W) float (with NaN) -> (3,H,W) uint8 via per-band 2/98 stretch.

    Returns (uint8 image, valid_mask) where valid_mask is False on NaN pixels.
    """
    valid = np.isfinite(img).all(axis=0)
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[0]):
        band = img[c]
        finite = band[np.isfinite(band)]
        lo, hi = np.percentile(finite, 2), np.percentile(finite, 98)
        b = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)
        b[~np.isfinite(band)] = 0.0
        out[c] = b * 255.0
    return out.astype(np.uint8), valid


def normalize(tile_u8: np.ndarray) -> np.ndarray:
    """(3,h,w) uint8 -> (3,h,w) float32 ImageNet-normalized."""
    x = tile_u8.astype(np.float32) / 255.0
    return (x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]


def make_blend_kernel(size: int) -> np.ndarray:
    """2D Hann window so overlapping tiles blend seamlessly (no seams)."""
    w = np.hanning(size)
    k = np.outer(w, w).astype(np.float32)
    return np.clip(k, 1e-3, None)


def infer_large_image(model, image_path, output_path, tile_size=512, overlap=64,
                      batch_size=4, device="cuda"):
    with rasterio.open(image_path) as src:
        img = src.read().astype(np.float32)  # (3, H, W)
        profile = src.profile

    img_u8, valid = stretch_to_uint8(img)
    H, W = img_u8.shape[-2:]
    stride = tile_size - overlap

    tiles, coords = [], []
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y2, x2 = min(y + tile_size, H), min(x + tile_size, W)
            tile = img_u8[:, y:y2, x:x2]
            pad = np.zeros((3, tile_size, tile_size), dtype=np.uint8)
            pad[:, : y2 - y, : x2 - x] = tile
            tiles.append(normalize(pad))
            coords.append((y, x, y2, x2))

    canvas = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)
    kernel = make_blend_kernel(tile_size)

    model.eval()
    for i in range(0, len(tiles), batch_size):
        batch = np.stack(tiles[i : i + batch_size])
        bt = torch.from_numpy(batch).to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits = model(bt)  # (B, 2, ts, ts)
            probs = torch.softmax(logits, dim=1)[:, 1]  # road class
        probs = probs.float().cpu().numpy()
        for j, (y, x, y2, x2) in enumerate(coords[i : i + batch_size]):
            h, w = y2 - y, x2 - x
            canvas[y:y2, x:x2] += probs[j, :h, :w] * kernel[:h, :w]
            weight[y:y2, x:x2] += kernel[:h, :w]

    canvas /= np.clip(weight, 1e-6, None)
    binary = (canvas > 0.5).astype(np.uint8)
    binary[~valid] = 0  # nodata -> background, never spurious road

    profile.update(count=1, dtype="uint8", compress="lzw", nodata=None)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(binary, 1)

    road_frac = binary.mean()
    print(f"wrote {output_path}  shape={binary.shape}  road fraction={road_frac:.4f}")
    return binary


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    name = cfg.get("model", {}).get("name", "nvidia/segformer-b2-finetuned-ade-512-512")
    num_labels = cfg.get("model", {}).get("num_labels", 2)
    model = build_segformer(name, num_labels).to(device)
    model.load_state_dict(ckpt["model"])
    return model


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input", default="data/inference/jaipur/jaipur_s2_rgb.tif")
    ap.add_argument("--output", default="data/inference/jaipur/predicted_mask.tif")
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    infer_large_image(model, args.input, args.output, args.tile_size,
                      args.overlap, args.batch_size, device)
