"""Phase 1.7 smoke tests: preprocessing + integration-contract compliance.

Uses a dummy 2-class model so it runs on CPU without the real checkpoint, while
still exercising the tiling, blending, NaN handling, and GeoTIFF writer.
"""
import numpy as np
import rasterio
import torch
from rasterio.transform import from_bounds

from src.inference import (stretch_to_uint8, make_blend_kernel, infer_large_image)


def test_stretch_handles_nan():
    img = np.random.uniform(200, 5000, (3, 40, 40)).astype(np.float32)
    img[:, 0, 0] = np.nan
    u8, valid = stretch_to_uint8(img)
    assert u8.dtype == np.uint8
    assert u8.max() <= 255 and u8.min() >= 0
    assert valid[0, 0] == False and valid[20, 20] == True


def test_blend_kernel_positive():
    k = make_blend_kernel(64)
    assert k.shape == (64, 64)
    assert (k > 0).all()


class _DummySeg(torch.nn.Module):
    """Returns 2-class logits at input resolution; predicts road on a diagonal."""
    def forward(self, x):
        b, _, h, w = x.shape
        logits = torch.zeros(b, 2, h, w)
        eye = torch.eye(min(h, w)).bool()
        logits[:, 1][:, :min(h, w), :min(h, w)][:, eye] = 10.0
        return logits


def test_inference_contract(tmp_path):
    # synthetic input GeoTIFF: float64, EPSG:4326, with a NaN pixel
    H, W = 300, 260
    data = np.random.uniform(200, 5000, (3, H, W)).astype(np.float64)
    data[:, 5, 5] = np.nan
    transform = from_bounds(75.78, 26.85, 75.87, 26.95, W, H)
    in_path = tmp_path / "in.tif"
    out_path = tmp_path / "predicted_mask.tif"
    with rasterio.open(in_path, "w", driver="GTiff", height=H, width=W, count=3,
                       dtype="float64", crs="EPSG:4326", transform=transform) as dst:
        dst.write(data)

    infer_large_image(_DummySeg(), str(in_path), str(out_path),
                      tile_size=128, overlap=32, batch_size=2, device="cpu")

    with rasterio.open(out_path) as src, rasterio.open(in_path) as ref:
        out = src.read(1)
        # --- integration contract ---
        assert src.count == 1
        assert out.dtype == np.uint8
        assert set(np.unique(out)).issubset({0, 1})
        assert src.crs == ref.crs
        assert src.transform == ref.transform
        assert src.profile["compress"].lower() == "lzw"
        # NaN pixel must be background
        assert out[5, 5] == 0
        # the diagonal road should have produced some road pixels
        assert out.sum() > 0
