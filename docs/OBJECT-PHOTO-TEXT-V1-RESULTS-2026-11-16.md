# Forge photo-text object branch v1 — 2026-11-16 UTC

Status: research only; no checkpoint is promoted or integrated into Android.

Run: https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/43804084493
Training commit: 6c75fbfb769a27591cdc27a9944d1bfe3232bc89

This run generated a fixed CC0-1.0 synthetic dataset of 960 objects across six procedural categories, with four rendered views each (3,840 images total). It used three fresh random initializations, 60 epochs, object-disjoint train/validation/test splits, a geometry-aware target mask, a training-median baseline, and Blender-rendered silhouette scoring.

| Seed | Text class acc / MAE | Single view acc / MAE | Four views acc / MAE | Image + text acc / MAE | Median baseline MAE | Proxy silhouette IoU | Candidate checkpoint SHA-256 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 43 | 1.000 / 0.207 | 0.854 / 0.217 | 0.885 / 0.215 | 1.000 / 0.209 | 0.214 | 0.386 | 7e29a283bd6c23c2598ae79aaf4564472c118c098f348302a46b79d8753ef8f3 |
| 53 | 1.000 / 0.203 | 0.750 / 0.217 | 0.844 / 0.216 | 1.000 / 0.207 | 0.214 | 0.379 | 702568d55fb6a4bce9cefb17051e744e56094e15bfb528d2f981959536dd0dc1 |
| 73 | 1.000 / 0.217 | 0.906 / 0.216 | 0.927 / 0.214 | 1.000 / 0.225 | 0.214 | 0.376 | 08d460e838b13bf84c44c3952f4f1064063bbce8599541d5ce2d1785649a242c |

Text classification is not meaningful here because each prompt states its object category. Image-only classification averages about 0.84 for one view and 0.89 for four views on these six synthetic categories. Geometry error is approximately tied with the median baseline; multimodal MAE is inconsistent across seeds. Mean proxy silhouette IoU is about 0.380.

This supports a useful synthetic test pipeline, but it does not demonstrate reliable photo-to-3D dimension inference, arbitrary object generation, photorealism, production topology, or general competence. The candidates remain separate from Forge Seed language weights.

Artifacts include data, manifests, checkpoints, render masks, and evaluation:
https://github.com/tpylman90-ctrl/Forge-Core/releases/tag/untagged-cc87253486911dbc08f7
