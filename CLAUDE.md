# CLAUDE.md — Route Resilience

Authoritative conventions & ownership for this repo. Read alongside the build
blueprint. **Standing rule: respect ownership boundaries — flag and ask before
editing the other person's files.**

## What this is

Occlusion-robust road extraction (CV) + graph-theoretic criticality analysis
(graph), demoed on Jaipur. Two-half pipeline joined by a single file contract.

## Ownership

| Area | Owner | Paths |
|---|---|---|
| CV half | **AA** | `src/data/`, `src/models/`, `src/losses/`, `src/train.py`, `src/inference.py`, `src/eval/seg_metrics.py`, `configs/`, `tests/` |
| Graph half | **Amrit** | `src/graph/`, `src/eval/topo_metrics.py`, `dashboard/`, `scripts/make_mock_mask.py` |
| Shared | both | `CLAUDE.md`, `requirements.txt`, `.gitignore`, `scripts/check_gpu.py`, `README.md` |

## Integration contract (FIXED — do not change without telling the other owner)

`data/inference/jaipur/predicted_mask.tif`
- single-channel **uint8** GeoTIFF
- values strictly in **{0, 1}** (0 = background, 1 = road)
- **same CRS + transform** as the input `jaipur_s2_rgb.tif`
- written with `compress="lzw"`

The CV half produces it; the graph half consumes it. Until it exists, the graph
half develops against `scripts/make_mock_mask.py` output (`mock_mask.tif`),
which has identical shape/CRS.

## Demo target

- City: Jaipur (Walled City + Raja Park + JLN Marg). BBox `[75.78, 26.85, 75.87, 26.95]`.
- Input: `data/inference/jaipur/jaipur_s2_rgb.tif` (Sentinel-2 SR, 10 m/px, EPSG:3857).
- Train data: DeepGlobe Road Extraction (6226 paired tiles, 0.5 m/px). **Actual
  paired-tile path is `data/deepglobe/train/archive/train`** (deeper than the
  blueprint's `data/deepglobe/train`; `archive/valid` + `archive/test` have no
  masks). Use this as `data.root_dir` in configs.

## Environment

- Python 3.12, torch 2.11.0+cu128, **RTX 4060 Laptop (8 GiB)**. CUDA confirmed.
- 8 GiB VRAM is tight: SegFormer-B2 @ batch 8/512px will likely OOM. Default to
  batch 4 or fall back to B0/B1 if needed (see risk register in the blueprint).
- Install: `pip install -r requirements.txt` — do **not** reinstall torch (would
  clobber the CUDA build).

## Conventions

- Run modules as packages from repo root: `python -m src.train ...`,
  `python -m src.inference ...`. `src/` is a package (`__init__.py` present).
- Configs are YAML in `configs/`. Don't hardcode hyperparameters in code that a
  config already owns.
- Training outputs go to `runs/<timestamp>_<tag>/` (config snapshot + `train.log`
  + `best.pth`/`last.pth`). `runs/` and `data/` are gitignored.
- Seed everything with 42 for reproducibility.
- Smoke-test every component on dummy/subset data before the real run.
- Commit after each working phase; use the phase name (e.g. `Phase 1.4: clDice loss`).
- 30-hour hackathon: working demo first, no refactoring for elegance, no new
  deps without asking.

## Status

- Phase 0: scaffolding done. **Data not yet on disk** — drop `jaipur_s2_rgb.tif`
  into `data/inference/jaipur/` and DeepGlobe into `data/deepglobe/train/`.
