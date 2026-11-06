# Approved Forge Core isolated multi-lane training — night run v1

Start three independent CPU learning jobs concurrently: language, Blender, and
Godot. Give each lane up to 3,600 seconds of training time, for up to three
runner-hours total. Use only standard GitHub-hosted Ubuntu runners in this
public repository; do not request a GPU, larger runner, external paid service,
or private data.

All lanes must initialize from the exact verified Forge Seed v0.7 checkpoint
from Actions run 42143390701:
- parent step: 95,000
- parent SHA-256: a7bcdb996e440f44d7496b887b54d19324927b4296fe9ed2f0f0293c88c63a0a
- tokenizer SHA-256: 4ecd7803d4c369e80bda2e74c30cfb769b7e3a942f931df83383eec9f794f04c

Isolation rules:
- Language continues the same pinned TinyStories split and resumes its own
  optimizer and RNG state.
- Blender and Godot each use separate full-weight research branches from the
  shared frozen parent, a fresh optimizer, separate official documentation
  pages, independent train/validation/test splits, and their own original
  lesson examples.
- No lane writes back to or merges into the shared parent or another lane.
- Keep all data manifests, source hashes, attribution, optimizer state, RNG
  state, checkpoint lineage, and evaluation outputs with the lane artifact.
- Keep every candidate research-only and do not update Android model weights.

Blender documentation is pinned to Blender 5.4 Manual prose (CC BY-SA 4.0);
the extraction excludes manual code blocks, and original lesson code is
CC0-1.0. Godot documentation is pinned to Godot 4.7 docs (CC BY 3.0).
For Blender and Godot, save ungraded responses to held-out task prompts and
report document token-loss separately from task competence. Do not certify
tool ability from token loss or unreviewed samples.

Preserve each successful lane as a separate Actions artifact and unpublished
draft release. Report the actual runtime, update count, source/version hashes,
matched held-out loss, task samples, capability limitations, and whether any
lane stopped before its time cap.
