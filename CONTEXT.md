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

### Available masks (in repo) — which to use

| File | Source | CRS | Density | Use for |
|---|---|---|---|---|
| `predicted_mask.tif` | Sentinel-2 10 m | EPSG:4326 | sparse (arterials only) | the strict integration-contract file |
| **`predicted_mask_fullcity.tif`** | **high-res ~1.2 m (proof-of-concept)** | **EPSG:3857** | **dense, full network** | **RECOMMENDED for graph dev + demo** |

**Amrit: build against `predicted_mask_fullcity.tif`** — it's a dense, realistic
city-wide network (skeletonize/heal/centrality will produce meaningful results,
not artifacts of a sparse mask). Read CRS+transform *from the file* (it's 3857,
not 4326). Both masks are uint8 {0,1}, LZW. The high-res one is from Esri
basemap tiles (proof-of-concept that the model scales with resolution); the
final submission will regenerate it from ISRO Cartosat-3/LISS-IV imagery via
the same `src/inference.py` (no code change — see §6b).

---

## 3. Ownership

| Area | Owner | Paths |
|---|---|---|
| CV half | **AA** | `src/data/`, `src/models/`, `src/losses/`, `src/train.py`, `src/inference.py`, `src/eval/seg_metrics.py`, `configs/`, `tests/` |
| Graph half | **Amrit** | `src/graph/`, `src/eval/topo_metrics.py`, `dashboard/`, `scripts/make_mock_mask.py` |
| Shared | both | `CLAUDE.md`, `CONTEXT.md`, `requirements.txt`, `.gitignore`, `scripts/check_gpu.py`, `README.md` |

**Standing rule:** flag and ask before editing the other person's files.

---

## 4. Current status — BOTH HALVES DONE & INTEGRATED

End-to-end pipeline works: satellite image -> road mask -> graph -> criticality
-> live disaster console. Integrated on the **`integration`** branch (CV `main`
+ Amrit's `amrit/graph-pipeline` merged).

### CV half (AA) — complete
- Phases 1.1–1.7 built & tested (**test suite: 15 passed, 1 skipped**).
- **Training trained to epoch 35** (converged): best **val IoU 0.5217 / Dice 0.6857**
  (`runs/20260627_032356_segformer_b2/best.pth`, local/gitignored).
- `src/inference.py` produces contract-compliant masks; supports `--upscale`,
  `--bands` (multi-sensor). **Key finding:** accuracy scales with resolution —
  Sentinel-2 (10 m) recovers only arterials (~12% recall); high-res (~1.2 m)
  recovers the full network.

### Graph half (Amrit) — complete (merged)
- `src/graph/{pipeline,centrality,osm_heal,run}.py`, `src/eval/topo_metrics.py`,
  `dashboard/build_map.py`. Skeleton -> graph -> Union-Find/MST healing ->
  betweenness centrality + ablation -> Folium dashboard.
- On the dense mask: 10 components, 98.9% in largest, 44 healed edges.
- ⚠️ **Known bug in `src/graph/pipeline.py:_dist_m`** — applies `np.radians()`
  AND `×111320`, so all distances (incl. `avg_path_length_m`) are **~57× too
  small**. One-line fix: `(lat2-lat1)*111320` (drop the radians). Important for
  the metrics table.

### Integrated system (new)
- **`dashboard/jaipur_console.html`** — self-contained Disaster Decision Console:
  occlusion-robust roads -> graph -> **Route Survival Score**, survival-optimal
  routing, **CartoDEM-driven multi-hazard sim** (flood / earthquake / landslide),
  live resilience index + junctions/residents/hospitals cut off. No server.
- `dashboard/jaipur_fullcity_criticality.html` — graph run on the dense mask.
- `Route_Resilience_BAH2026_Final.pptx` — idea-submission deck (maps to the
  official PUB template prompts). Diagrams in `results/diagram_*.png`.
- New deps: `sknw` (graph), and dev-only: `python-pptx`, `selenium`, `matplotlib`.

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

# run the graph half on a mask -> dashboard HTML + metrics JSON
#   (needs sknw; PYTHONUTF8=1 avoids a Windows unicode-print crash;
#    the graph code expects EPSG:4326 — reproject 3857 masks first)
PYTHONUTF8=1 python -m src.graph.run \
  --mask data/inference/jaipur/predicted_mask_fullcity_4326.tif \
  --out  dashboard/jaipur_fullcity_criticality.html

# open the live console (self-contained, no server)
#   dashboard/jaipur_console.html  -> just open in a browser
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

**Code / correctness:**
1. Fix the `_dist_m` 57× bug in `src/graph/pipeline.py` (graph half) and re-run
   so the metrics table is accurate.
2. Train the U-Net baseline briefly for the comparison number (IoU vs SegFormer).
3. Occlusion-recall metric: hand-annotate ~20 tiles' occluded regions.

**Demo / pitch:**
1. Record a ~30 s GIF of `dashboard/jaipur_console.html` (flood slider, bridge
   cut, survival routing) for slide 6 of the deck.
2. Fill team name + 2 member rows in `Route_Resilience_BAH2026_Final.pptx`;
   optionally render the Mermaid process/architecture diagrams in.
3. On event day: re-run inference on **Cartosat-3 / LISS-IV** imagery (no code
   change) for a high-res, indigenous-data demo.

**Git:** work is on the **`integration`** branch. Open a PR to `main` when the
team has reviewed (CV `main` history is currently clean / CV-only).
