# Blender/Godot answer-label probe audit — 2026-11-09

This was an evaluation-only rerun of the archived Blender/Godot v2 parents and selected checkpoints from [training run 42541109934](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42541109934). No weight or optimizer updates occurred. The parent and selected checkpoints were identical in both lanes because v2 early-stopped without validation improvement.

The audit reports the original full-answer-text ranking and a separate next-label score. The label score ranks the next-token likelihood for A/B/C/D after the full question and options. Four answer rotations per base question expose answer-position consistency. This remains a small forced-choice diagnostic; it does not test whether Forge can produce or execute useful Blender/Godot work.

| Lane | Parent / selected full-answer ranking | Parent / selected direct label score | All four rotations correct (parent / selected) |
|---|---:|---:|---:|
| Blender | 0/72 / 0/72 | 18/72 / 18/72 (25.0%, chance baseline) | 0 / 0 |
| Godot | 17/92 / 17/92 (18.5%) | 24/92 / 24/92 (26.1%) | 0 / 0 |

The direct-label score confirms that the earlier low text-ranking totals should not be read as a clean competence measurement. Label selection is at chance for Blender and only slightly above chance for Godot; neither lane consistently solved any base question across all four answer positions. Since the checkpoints are identical, this audit says nothing about whether training changed skill.

The source and pinned eight-page holdout manifests, both checkpoints, per-item outputs, and full scores are preserved in the [evaluation-only audit release](https://github.com/tpylman90-ctrl/Forge-Core/releases/tag/untagged-847b63d1144a559e7132). The next tool-lane work should improve the benchmark with held-out code/scene tasks checked by free Blender/Godot headless runs and small human review, before another specialist continuation is judged against this forced-choice score.
