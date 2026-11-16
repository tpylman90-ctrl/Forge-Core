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
