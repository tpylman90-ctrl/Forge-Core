# Approved conservative Forge Seed continuation

Run one bounded, uniform TinyStories language continuation from the exact verified
95,000-step Forge Seed checkpoint in run 42143390701.

- Resume Forge's original scratch-origin weights, AdamW moments, RNG states,
  tokenizer, and pinned TinyStories training split.
- Keep Blender and Godot as separate lanes; do not mix their data into this run.
- Reduce the learning rate from the parent's 0.0003 to 0.0001 while preserving
  the saved optimizer moments.
- Run at most 4,000 updates, checking the same 2,048-window validation sample
  every 1,000 updates. Stop after two consecutive checks without at least a
  0.0001 validation-loss improvement over the best checkpoint so far.
- Select checkpoints using validation only. Compare the parent, selected
  checkpoint, and last attempted checkpoint on the same untouched pinned
  TinyStories range using 2,048 evenly spaced 128-token windows.
- Preserve complete parent and candidate checkpoints, optimizer state, RNG
  state, exact source data, hashes, evaluation curves, capability diagnostics,
  and lineage in Actions artifacts and an unpublished draft release.
- Mark all outputs research-only. Do not merge teacher weights, promote a
  candidate, or integrate any checkpoint into Android.

The goal is a cautious actual Forge weight update with explicit rollback if the
validation signal does not repeat. Next-token loss and toy forced-choice probes
must not be described as proof of comprehension, reasoning, or capability gain.


## Reproducible retry

The first seeded attempt completed all 4,000 updates and evaluation but stopped before artifact upload because the parent evaluation directory lacked `source-manifest.json`. The workflow now includes that manifest. Repeat this exact deterministic run from the same verified parent and preserve the full artifact and draft archive.
