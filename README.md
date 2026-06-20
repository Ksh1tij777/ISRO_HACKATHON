# Route Resilience

Occlusion-robust road extraction + graph-theoretic criticality analysis,
demoed on Jaipur. ISRO/NNRMS hackathon submission.

Two halves joined by one file contract (`data/inference/jaipur/predicted_mask.tif`):
1. **CV** — SegFormer + clDice loss extracts roads from Sentinel-2, robust to
   tree/shadow occlusion.
2. **Graph** — skeletonize → heal → centrality/ablation to find "gatekeeper"
   roads whose loss most degrades the network.

## Quickstart

```bash
pip install -r requirements.txt   # torch already installed w/ CUDA — don't reinstall
python scripts/check_gpu.py       # confirm GPU is visible
```

Then drop the data in place:
- `data/inference/jaipur/jaipur_s2_rgb.tif` (Sentinel-2 RGB)
- `data/deepglobe/train/` (DeepGlobe pairs, for training)

See `CLAUDE.md` for ownership, conventions, and the integration contract, and
the build blueprint for the phase-by-phase plan.

## Status

Phase 0 (scaffolding) complete. CV pipeline (Phase 1) next.
