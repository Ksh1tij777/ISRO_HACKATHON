# Route Resilience — Project Context & Handoff

Shared reference for the whole team. Read this first to understand *what exists,
how to run it, and what's next*. For ownership rules & the integration contract,
see `CLAUDE.md`. For the full phase-by-phase plan, see the build blueprint.

---

## 1. What this project is

Occlusion-robust road extraction (CV) + graph-theoretic criticality analysis
(graph), demoed on **Jaipur**. Two halves joined by **one file**:

```
[CV half] satellite image --> predicted_mask.tif --> [Graph half] road network analysis
```

- **CV half (AA):** SegFormer + clDice loss extracts roads from Sentinel-2,
  robust to tree/shadow occlusion.
- **Graph half (Amrit):** skeletonize the mask -> graph -> heal gaps ->
  centrality/ablation -> find "gatekeeper" roads -> interactive dashboard.

---

## 2. The integration contract (FIXED)

`data/inference/jaipur/predicted_mask.tif`
- single-channel **uint8** GeoTIFF, values strictly **{0, 1}** (0=bg, 1=road)
- **same CRS + transform** as the input `jaipur_s2_rgb.tif`
- written with `compress="lzw"`

CV produces it; Graph consumes it. Until it's final, the graph half develops
against `scripts/make_mock_mask.py` output (identical shape/CRS).

---

## 3. Ownership

| Area | Owner | Paths |
|---|---|---|
| CV half | **AA** | `src/data/`, `src/models/`, `src/losses/`, `src/train.py`, `src/inference.py`, `src/eval/seg_metrics.py`, `configs/`, `tests/` |
| Graph half | **Amrit** | `src/graph/`, `src/eval/topo_metrics.py`, `dashboard/`, `scripts/make_mock_mask.py` |
| Shared | both | `CLAUDE.md`, `CONTEXT.md`, `requirements.txt`, `.gitignore`, `scripts/check_gpu.py`, `README.md` |

**Standing rule:** flag and ask before editing the other person's files.

---

## 4. Current status (as of pause)

### CV half — Phases 1.1–1.7 BUILT & TESTED
| Phase | Deliverable | Status |
|---|---|---|
| 1.1 | Dataset + occlusion augmentations | done, tested |
| 1.2 | U-Net baseline | done (not yet trained for the comparison number) |
| 1.3 | SegFormer wrapper | done, tested |
| 1.4 | dice + bce + clDice losses | done, tested |
| 1.5 | Training loop (+ `--resume`) | done; **training PAUSED at epoch 8** |
| 1.6 | Eval metrics (incl. occlusion-recall) | done, tested |
| 1.7 | Tiled Jaipur inference | done, tested (contract-compliant) |

- **Test suite: 15 passed, 1 skipped.**
- **Best model so far:** epoch 8, **val IoU 0.4736 / Dice 0.6428**.
- Checkpoints (local only, gitignored): `runs/20260622_020144_segformer_b2/{best,last}.pth`.

### Graph half — Phase 2 NOT STARTED (Amrit)
Skeleton->graph, healing, centrality/ablation, topo metrics, dashboard.

---

## 5. How to run things

```bash
# one-time setup
pip install -r requirements.txt          # torch already CUDA-installed; don't reinstall
python scripts/check_gpu.py              # confirm GPU

# run the tests
python -m pytest -q

# train (fresh)
python -m src.train --config configs/segformer_b2.yaml --tag segformer_b2

# RESUME the paused run (continues from epoch 9 -> 40)
python -m src.train --config configs/segformer_b2.yaml --tag segformer_b2 \
  --resume runs/20260622_020144_segformer_b2/last.pth

# produce the handoff mask once a good checkpoint exists
python -m src.inference \
  --checkpoint runs/<run>/best.pth \
  --input  data/inference/jaipur/jaipur_s2_rgb.tif \
  --output data/inference/jaipur/predicted_mask.tif
```

---

## 6. Data facts (IMPORTANT — deviate from the blueprint)

- **Train data path:** `data/deepglobe/train/archive/train` (deeper than the
  blueprint's `data/deepglobe/train`; 6226 paired tiles. `archive/valid` +
  `archive/test` have no masks).
- **Jaipur GeoTIFFs** (`data/inference/jaipur/`):
  - CRS is **EPSG:4326** (lat/lon), NOT 3857. Bbox `[75.78, 26.85, 75.87, 26.95]`.
  - **float64** reflectance, shape (3, 1115, 1002), with **~0.09% NaN** nodata
    pixels. `inference.py` handles stretch + NaN; graph half: ignore value-0 nodata.
  - `jaipur_s2_nir.tif` is a false-color **NRG** composite (band1=true NIR).
- `data/` and `runs/` are **gitignored** — large files never get committed.

---

## 6b. Sensor / resolution strategy (IMPORTANT)

- **OSM is ground-truth/benchmark ONLY** — used as training labels and for the
  Topological-Accuracy eval. It is NEVER fed in as the road network (that would
  be circular and against the rules).
- **Resolution matters more than the model.** The DeepGlobe-trained model is built
  for high-res (~0.5 m). On **Sentinel-2 (10 m)** roads are ~1 px, so extraction is
  sparse — this is a property of the data, not a model failure. The challenge
  provides better imagery: **LISS-IV (5.8 m, open)** and **Cartosat-3 (high-res,
  given during the event for evaluation)**. Run the demo on those.
- **Switching sensors = swap the .tif**, no retraining. `src/inference.py` now
  supports per-sensor input handling:
  - `--upscale N` — only needed for coarse input (use 4 for S2, 1 for hi-res).
  - `--bands a,b,c` — map sensor bands into the model's RGB slots. S2 RGB = default
    `1,2,3`. LISS-IV is G/R/NIR (no blue) -> pick a false-colour mapping.
    Panchromatic (1 band) is auto-replicated to 3.

## 7. Environment

Python 3.12, torch 2.11.0+cu128, **RTX 4060 Laptop (8 GiB)**, CUDA OK.
Training: batch 4, ~5.8 min/epoch, full 40-epoch run ≈ 4h, peak VRAM ~3.7 GiB.

---

## 8. What's next

**CV (AA):**
1. Resume training to epoch 40 (one command above).
2. Re-run inference with the final model -> hand `predicted_mask.tif` to Amrit.
3. Train U-Net baseline briefly for the comparison number.
4. Hand-annotate ~20 tiles' occluded regions for the occlusion-recall metric.

**Graph (Amrit):**
1. Build Phase 2.1–2.6 (mock mask -> graph -> heal -> centrality -> dashboard).
2. Develop against `mock_mask.tif` until the real mask lands.

**Joint (Phase 3–4):**
Integrate real mask, tune healing, build metrics tables + before/after visuals,
5-slide deck, pre-recorded demo GIF.
