# Guided specialist continuation v5 — 2026-11-17

## Why this run follows v4
- Blender and Godot each improved held-out documentation loss in all three seeds after v3→v4; both stayed under the +0.05 TinyStories retention guard.
- The v4 language lane selected its parent in all seeds, so this run does not spend more updates on the flat language arm.
- Earlier concentrated surprise replay and source-oracle traces hurt the core language validation loss substantially. Keep replay balanced and uniform within each source.
- Visual branch v2 improved mean synthetic Blender silhouette IoU from 0.380 to 0.475 across three fresh-initialized seeds, but remains below 0.5 and its six category labels are too easy to support an open-world claim. Do not combine this unrelated fresh-init vision model with Forge Seed or count it as application competence.
- Use v4 specialist artifacts as exact parents, including optimizer and sample RNG state; halve the learning rate to test whether continued domain gains can be obtained with lower drift.

## Controlled run
- Six candidates: Blender and Godot × seeds 43, 53, 73.
- Parent: matching seed/lane candidate from run 43804101939; verify artifact, metrics, checkpoint SHA-256, seed, lane, optimizer state and sampler RNG before training.
- Three progressive 500-update stages per candidate; validation every 100 updates; use 50% pinned TinyStories replay for each specialist batch; learning-rate scale 0.5.
- Reuse pinned license-checked manuals and the exact TinyStories revision and tokenizer from v4.
- Select by held-out validation, report test once after selection, and reject any candidate beyond the existing +0.05 core-language guard.
- Preserve parent and candidate checkpoint hashes, optimizer state, RNG state, stage hashes, source manifest, data, curves and evaluations in a draft research archive.

## Interpretation
A win requires lower domain test loss than the matching v4 parent across seeds without violating the core-language guard. This only shows better next-token prediction on the pinned documentation. It does not certify Blender/Godot task skill. Language training is deferred pending a stronger independent evaluation and a broader reviewed corpus; visual training remains a separate random-init research branch. No model is merged, promoted to Android, or used with personal memory.

Implementation note: Initial attempt 44006678996 stopped during pre-training validation because the v4 artifact predates the lane-update bookkeeping fix. No optimizer steps ran. Retry after the check was corrected to verify the artifact hash, lane/seed, optimizer and sampler RNG without assuming the newer cumulative-update field.
