# Forge object photo-text branch v1 run

Run the separate fresh-random-init object branch experiment.

- Generate the CC0 procedural dataset locally in Blender, with hashes and fixed splits.
- Train three seeds from random initialization; do not load Forge Seed language weights.
- Evaluate text-only, single-view, four-view, and combined inputs on the object-disjoint test split.
- Render predicted against target proxy geometry and record silhouette IoU.
- Preserve generated data, model/optimizer checkpoints, hashes, metrics, Blender version, and draft research archive.
- Keep the experiment research-only; do not merge weights, integrate Android, or claim general photo-to-3D ability from this narrow synthetic benchmark.

## Retry

Restart after installing Blender's required EGL and headless OpenGL libraries; retain the same dataset seed and training settings.

## Retry 2

Restart after adding the missing dataset mode switch; retain the same code-level dataset seed and three model seeds.

## Retry 3

Use 128 training, 16 validation, and 16 test instances per category; train 60 epochs; compute constant-median parameter baselines; score Blender PNG masks with Pillow and fail if masks are empty or nearly full.


## Follow-up run — geometry-target correction

Repeat the fixed CC0 dataset and three-seed study after correcting the class-prior baseline and masking parameter loss/metrics to geometric controls that are actually used by each object category. Unused controls, including p7, must not distort training or validation selection. Preserve all prior modality reports and Blender silhouette scoring; keep every checkpoint research-only and isolated from the Forge Seed language weights.
