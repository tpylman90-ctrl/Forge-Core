# Forge photo-text object branch v4 — results (2026-11-17)

## Run and data

Run: [GitHub Actions 44010088887](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/44010088887)  
Draft archive: [CC0 data, hashes, checkpoints, optimizer state and render evaluations](https://github.com/tpylman90-ctrl/Forge-Core/releases/tag/untagged-5d9eb8249999ed556969)

This branch starts from fresh random initialization and does not load Forge Seed language weights. The CC0-1.0 dataset is generated from repository Blender geometry: 960 objects across six categories, with 768 train, 96 validation and 96 test objects. Each object has four RGB views and four semantic part maps.

All categories appear in training. Half of each category's test examples are in-range controls; the other half hold out extreme proportion/detail controls outside the training range (training uses normalized p0-p6 values 0.20–0.80; held-out examples push one control to 0.05–0.15 or 0.85–0.95). Three seeds (43, 53, 73) trained 60 epochs each from scratch.

## Results

| Seed | Single-view geometry MAE seen / held-out | Four-view MAE seen / held-out | Part mIoU seen / held-out | Silhouette IoU seen / held-out | Training-median MAE |
|---:|---:|---:|---:|---:|---:|
| 43 | 0.139 / 0.179 | 0.139 / 0.179 | 0.271 / 0.268 | 0.569 / 0.432 | 0.163 |
| 53 | 0.144 / 0.180 | 0.143 / 0.179 | 0.276 / 0.279 | 0.539 / 0.422 | 0.163 |
| 73 | 0.139 / 0.179 | 0.138 / 0.180 | 0.271 / 0.276 | 0.570 / 0.441 | 0.163 |
| **Mean** | **0.141 / 0.179** | **0.140 / 0.179** | **0.273 / 0.274** | **0.559 / 0.432** | **0.163** |

The combined image-plus-text control MAE averaged 0.116 on seen examples and 0.163 on held-out examples. The image-only held-out geometry MAE was about 0.179, above the 0.163 training-median baseline; the combined result only matched that baseline. Text prompts include the category and approximate dimensions, so the combined result does not prove that photos alone recover geometry. Four views barely changed geometry MAE relative to one view.

The held-out part mIoU stayed near 0.274, and the proxy silhouette IoU averaged 0.432 on held-out extremes versus 0.559 on in-range examples. The family split is a useful controlled shape-transfer check, but the held-out shape is still noticeably less accurate. Silhouette IoU is a Blender proxy metric under target-fit camera framing; it is not a production mesh or photo-to-3D quality score.

## Interpretation and next gate

This is a promising, repeatable synthetic part/shape signal across seeds, not arbitrary photo-to-3D competence. It improves on the earlier whole-category holdout in the sense that the model can be evaluated on new parameter combinations within known categories. Its held-out geometry regression does not beat the simple median predictor from images alone.

For the next run, keep held-out controls and source manifests fixed, make prompt dimensions optional or noisy in a separate modality arm, add varied camera/light/background conditions, and score absolute scale separately from shape proportions. Continue per-seed checkpoints and use fresh, independently authored or licensed reference geometry before claiming transfer beyond this procedural Blender family.

No candidate is promoted or integrated into Android.
