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

import cv2
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


def select_bands(img: np.ndarray, bands=None) -> np.ndarray:
    """Map a sensor's bands into the model's 3 RGB-like input channels.

    The model expects RGB, but sensors differ:
    - Sentinel-2 RGB  -> bands 1,2,3 (default)
    - LISS-IV (G,R,NIR, no blue) -> e.g. --bands 2,1,1 or 3,2,1 (false colour)
    - Cartosat-3 panchromatic (1 band) -> auto-replicated to 3 channels

    bands: list of 1-indexed band numbers (length 3), or None for first 3.
    """
    C = img.shape[0]
    if bands is None:
        bands = [1, 2, 3] if C >= 3 else [1, 1, 1]  # panchromatic -> replicate
    if len(bands) != 3:
        raise ValueError(f"--bands needs exactly 3 values, got {bands}")
    idx = [b - 1 for b in bands]
    if max(idx) >= C:
        raise ValueError(f"band index {max(idx)+1} out of range (image has {C} bands)")
    return np.stack([img[i] for i in idx])


def infer_large_image(model, image_path, output_path, tile_size=512, overlap=64,
                      batch_size=4, device="cuda", upscale=1, bands=None):
    with rasterio.open(image_path) as src:
        img = src.read().astype(np.float32)  # (C, H, W)
        profile = src.profile

    img = select_bands(img, bands)  # -> (3, H, W) in RGB order
    img_u8, valid = stretch_to_uint8(img)
    H0, W0 = img_u8.shape[-2:]  # native size (output is written at this size)

    # Scale-matching for low-res input (e.g. Sentinel-2 10 m/px): upsampling so
    # roads occupy a pixel-width closer to the model's training data improves
    # recall. It adds no information, only rescales feature size. x4 is a good
    # default for S2; x6+ over-zooms and the model loses big-road context.
    if upscale != 1:
        img_u8 = np.stack([
            cv2.resize(img_u8[c], (W0 * upscale, H0 * upscale),
                       interpolation=cv2.INTER_CUBIC)
            for c in range(3)
        ])
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
    if upscale != 1:  # bring probability map back to native resolution
        canvas = cv2.resize(canvas, (W0, H0), interpolation=cv2.INTER_AREA)
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
    ap.add_argument("--upscale", type=int, default=1,
                    help="upsample factor before inference (4 recommended for "
                         "Sentinel-2 10m/px; scale-matches roads to training data)")
    ap.add_argument("--bands", default=None,
                    help="comma-separated 1-indexed bands -> RGB slots "
                         "(e.g. '3,2,1'). Default: first 3 bands; 1-band input "
                         "is auto-replicated. Use for LISS-IV/Cartosat etc.")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bands = [int(b) for b in args.bands.split(",")] if args.bands else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    infer_large_image(model, args.input, args.output, args.tile_size,
                      args.overlap, args.batch_size, device, args.upscale, bands)
