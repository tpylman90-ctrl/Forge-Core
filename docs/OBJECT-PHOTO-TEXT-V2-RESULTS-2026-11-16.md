# Forge photo-text object branch v2 — 2026-11-16 UTC

Status: research only. The branch starts from fresh random weights, does not load Forge Seed language weights, and is not integrated into Android.

Run: https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/43805362722
Workflow/code commit: 867759fcdcca9c250e57a5be21ff84de4bc0ec29

The fixed CC0-1.0 dataset contains 960 synthetic objects in six categories, each with four Blender-rendered views (3,840 images). V2 adds a foreground-mask input channel and masks the geometry loss/metric to controls actually used by each category. Each seed trained for 60 epochs (1,440 updates) from a fresh initialization and was tested on 96 object-disjoint held-out examples. Blender rendered candidate and target silhouettes for an additional check.

| Seed | Parameters | Text class acc / MAE | Single view acc / MAE | Four views acc / MAE | Image + text acc / MAE | Median baseline MAE | Blender silhouette IoU | Candidate checkpoint SHA-256 |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 43 | 162,566 | 1.000 / 0.185 | 1.000 / 0.211 | 1.000 / 0.204 | 1.000 / 0.173 | 0.214 | 0.467 | a92fde8002a0e59ab35a97067500475f1dd72e1f740c35f4f45e0941f83ddf1d |
| 53 | 162,566 | 1.000 / 0.161 | 1.000 / 0.199 | 1.000 / 0.195 | 1.000 / 0.152 | 0.214 | 0.484 | d4a99274b5d9d9163fc89ba3a3ad274f5e3cf3f2c20cc389023af2b50160a641 |
| 73 | 162,566 | 1.000 / 0.173 | 1.000 / 0.205 | 1.000 / 0.200 | 1.000 / 0.160 | 0.214 | 0.475 | e5f009faa7e2066e8e67292cd8a5c4b11170a2e67b95d7bbc55902addac0118d |

Normalized parameter MAE is below the per-dimension training-median baseline in all seeds. The mean silhouette IoU is 0.475, up from 0.380 in the matched v1 study. Text class accuracy is trivial because each prompt names its object category. Image-only category accuracy is 100% on this small six-category synthetic test, not an out-of-domain photo benchmark.

This is a repeatable gain on synthetic geometry and rendered silhouettes. IoU below 0.5 still indicates substantial shape mismatch. The study does not demonstrate arbitrary photo-to-3D generation, photorealism, production mesh topology, or general competence. No candidate is promoted.

The run artifacts preserve the CC0 dataset and hashes, training curves, checkpoints, parent data, render pairs, and scores. Draft archive:
https://github.com/tpylman90-ctrl/Forge-Core/releases/tag/untagged-e83d4d8ce1eef8d23c6b
