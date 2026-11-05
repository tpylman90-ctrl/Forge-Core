# Forge Seed v0.7 Core Language Continuation — 52,000 Updates

**Run date:** 2026-11-05  
**Status:** Completed successfully; research checkpoint retained, not promoted  
**Actions run:** [continuation 42118418883](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42118418883)  
**Parent run:** [44,000-step continuation 42116842094](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42116842094)

## Training result

This continued the original scratch-initialized Forge Seed language model from
step 44,000, resuming its AdamW optimizer and RNG states. It performed 8,000
more next-token updates on the exact same pinned TinyStories training and
validation split, with the same BPE tokenizer settings, 2.22M-parameter
architecture, seed 43, batch size 4, context 128 and learning rate 0.0003.

- Total steps: **52,000**
- Additional updates: **8,000**
- Runtime: **385.44 seconds**
- Same-protocol 64-window progress loss: **2.238001 at step 44,000 →
  2.201201 at step 52,000** (1.64% reduction).
- Cumulative 64-window curve change: **2.369578 at step 28,000 →
  2.201201 at step 52,000** (7.11% reduction).
- Separate final 256-window diagnostic: **2.226931**. It uses more windows,
  so it should not be compared directly with 64-window curve values.
- Sampled estimated bits per byte: **0.882384**, estimated from 256 token
  windows, not an exact full-validation measurement.

## Capability check

The separate forced-choice diagnostic scored **20/40 (50%)**; **0/20**
mirrored pairs had both answer orders correct. This remains chance-level on
the small template-based exam. Lower next-token loss is evidence of improved
held-out text prediction under this training setup, not demonstrated
improvement in comprehension, reasoning, instruction following or general
intelligence.

## Lineage and retained output

- Dataset revision: `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`
- Declared license: `CDLA-Sharing-1.0`
- Parent step: **44,000**, run 42116842094.
- Candidate step: **52,000**, run 42118418883.
- Candidate archive SHA256:
  `3cdbcd1fb46df899eb7260911959b07f6ce975c4915413c039fb8acd91a9c14f`
- Artifact: `forge-seed-v07-continuation`, retained through
  **2027-02-03**. The archive includes the parent/candidate checkpoint,
  exact train/validation files, source manifest, optimizer state, metrics,
  samples and checkpoint SHA256 lineage.

## Decision

The 52,000-step candidate continues the monotonic improvement in the
same-protocol 64-window progress curve. Continue in bounded increments while
this fixed held-out signal improves. Keep the weights research-only until
broader language quality and independent capability gates pass; do not
automatically replace Android weights.
