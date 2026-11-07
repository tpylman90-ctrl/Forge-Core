# Fresh Blender and Godot holdout evaluation

Evaluate, without training or changing weights:

- The verified Forge Seed v0.7 95k parent from run 42143390701.
- All four Blender and all four Godot checkpoints from the completed short sweep, run 42373188405.

Use the new version-pinned official documentation holdouts in `.github/workflows/fresh-tool-evaluation.yml` and `tool_capability_eval.py`. These are topic pages not present in the specialist training sources. Preserve the exact HTML/text hashes and attribution.

Score each checkpoint on fixed-window loss for the four unseen official prose pages and on the balanced answer-keyed four-choice tasks with four answer-position rotations. No holdout material enters training or checkpoint selection. Generated code samples remain ungraded; do not claim practical Blender/Godot competence, merge candidates, or promote any checkpoint.

This is a read-only evaluation of the existing research artifacts; no training run is triggered.
Retry evaluation after fixing lane-specific artifact names.

Retry the evaluation after correcting the Godot score paths as well.
