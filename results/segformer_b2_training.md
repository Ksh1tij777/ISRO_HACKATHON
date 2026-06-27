# SegFormer-B2 training run — results

Model: `nvidia/segformer-b2-finetuned-ade-512-512` (re-headed to 2 classes)
Data: DeepGlobe road extraction (5604 train / 622 val, seeded 90/10 split)
Loss: 0.5·Dice + 0.3·BCE + 0.2·clDice
Config: batch 4, lr 6e-5 (cosine + 5% warmup), AMP, 512px, seed 42
Hardware: RTX 4060 Laptop (8 GiB), ~6.9 min/epoch

## Best model
**Epoch 33 — val IoU 0.5217 / Dice 0.6857** (`best.pth`)

## Run notes
- Trained epochs 0–8, paused, resumed via `--resume`, continued 9–34.
- Stopped at epoch 35 (system interruption overnight) — model had converged
  (LR ~1.6e-6 by then), so the missing epochs 36–39 are negligible.
- IoU climbed steadily and plateaued ~0.52. clDice trades a little raw IoU for
  road connectivity; heavy occlusion augmentation lowers clean-val IoU but
  raises robustness (the project's actual goal).

## Per-epoch validation metrics
See `segformer_b2_metrics.csv`. Summary:

| Epoch | Val IoU | Val Dice |
|---|---|---|
| 0 | 0.3648 | 0.5346 |
| 5 | 0.4690 | 0.6385 |
| 8 (pause) | 0.4736 | 0.6428 |
| 14 | 0.5048 | 0.6710 |
| 20 | 0.5090 | 0.6746 |
| 26 | 0.5188 | 0.6831 |
| 28 | 0.5196 | 0.6839 |
| **33 (best)** | **0.5217** | **0.6857** |
| 34 (last) | 0.5188 | 0.6832 |

Full raw logs: `train_log_epoch0-8.log`, `train_log_epoch9-34.log`.

> Checkpoints (`best.pth`/`last.pth`, 328 MB each) are NOT in git — they live
> locally under `runs/20260627_032356_segformer_b2/`. Share directly if needed.
