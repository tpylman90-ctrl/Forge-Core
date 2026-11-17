# Forge photo-text object branch v3 — results (2026-11-17)

## Run and lineage

Run: [GitHub Actions 44008238525](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/44008238525)  
Code fix: [9c4122d](https://github.com/tpylman90-ctrl/Forge-Core/commit/9c4122df46d37268e57b4b18950fad3cb2348fc4)  
Draft research archive: [synthetic data, hashes, checkpoints and render pairs](https://github.com/tpylman90-ctrl/Forge-Core/releases/tag/untagged-e6660603ebbd8583a379)

The first attempt, run 44007365268, stopped before training because the trainer referenced an undefined family constant. No model updates occurred in that attempt. The constant was fixed and CI passed before this retry.

The v3 branch starts from fresh random initialization and loads no Forge Seed language weights. Its fixed synthetic dataset is declared CC0-1.0 and contains 672 objects: 512 train objects across crate, stool, lamp and hovercraft; 64 validation objects from those same four families; and 96 test objects, including 64 seen-family and 32 chair/mug held-out-family objects. Every object has four RGB views and four semantic part-mask views. Dataset manifest SHA-256: `fb822722cae5fff4abb31f6fb13924c0ef7e8f4cc5949567782f0a31d27706be`.

Three seeds (43, 53, 73) each trained for 60 epochs / 960 batches from random initialization. The study added a five-class semantic part head and reported seen-family and held-out-family measures separately.

## Results

| Seed | Combined parameter MAE | Training-median baseline MAE | All-test silhouette IoU | Seen-family IoU | Held-out chair/mug IoU |
|---:|---:|---:|---:|---:|---:|
| 43 | 0.209 | 0.214 | 0.313 | 0.364 | 0.211 |
| 53 | 0.195 | 0.214 | 0.353 | 0.387 | 0.284 |
| 73 | 0.197 | 0.214 | 0.355 | 0.408 | 0.250 |
| **Mean** | **0.200** | **0.214** | **0.340** | **0.386** | **0.248** |

Across seeds, combined-input foreground part mIoU averaged about 0.258. The held-out-family parameter MAE averaged about 0.217, roughly tied with or slightly worse than the overall training-median baseline. Seen-family parameter MAE averaged about 0.193.

The combined classifier's 0.667 overall accuracy consists of 1.0 on the four trained classes and 0.0 on chair/mug, which were never present in training or validation. It does not demonstrate zero-shot category recognition. The prompt explicitly names object classes and dimensions; text classification accuracy is not useful evidence of photo understanding.

## Interpretation

The semantic part masks support measurable part prediction on the synthetic families, but foreground part mIoU remains around 0.26. Proxy silhouette overlap is notably lower on withheld families (mean 0.248) than on seen families (0.386). This is a useful generalization failure diagnosis, not a capability gain. The v3 result cannot be compared directly with v2's 0.475 overall IoU because v3 includes entirely withheld categories and scores them separately. All models remain research-only and have not been promoted or integrated into Android.

## Next visual-data design

- Include a wider set of licensed or repository-generated object families and hard-surface assemblies.
- Hold out entire mesh/template families within trained categories for the main generalization score; reserve whole-category exclusion as a separate zero-shot stress test.
- Vary camera, light and background, while keeping exact geometry, part-map and source hashes.
- Use diverse prompt templates, including prompts that omit category names or dimensions, and score each modality independently.
- Compare silhouette and part metrics against explicit baselines, with human review of the resulting meshes before any capability claim.

Keep the visual branch separate from the language, Blender and Godot checkpoints. No personal memory is used.
