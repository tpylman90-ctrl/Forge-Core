# Forge Seed v0.7 Core Language Continuation — 44,000 Updates

**Run date:** 2026-11-05  
**Status:** Completed successfully; research checkpoint retained, not promoted  
**Actions run:** [continuation 42116842094](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42116842094)  
**Parent run:** [36,000-step continuation 42115007553](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42115007553)

## Training and language-model result

This was a true continuation of Forge Seed's original scratch-initialized
language model. It resumed the step-36,000 model, AdamW optimizer and RNG
state, then completed 8,000 next-token updates. The same pinned TinyStories
train/validation files, BPE tokenizer settings, 2.22M-parameter architecture,
seed 43, batch size 4, context 128 and learning rate 0.0003 were retained.

- Total optimizer steps: **44,000**
- Additional updates: **8,000**
- Runtime: **455.91 seconds**
- Same-protocol 64-window progress loss: **2.308715 at step 36,000 → 2.238001 at step 44,000** (3.06% reduction).
- Cumulative 64-window curve change: **2.369578 at step 28,000 → 2.238001 at step 44,000** (5.55% reduction).
- Separate final 256-window diagnostic: **2.262222**; it uses more windows and should not be directly compared to the 64-window curve.
- Sampled estimated bits per byte: **0.896367**; estimated from 256 token
  windows, not an exact full-validation BPE/byte measurement.

## Capability check

The template-based forced-choice diagnostic scored **20/40 (50%)**. Only
**1/20** mirrored pairs had both answer orders correct. This remains
chance-level evidence for that small test. The result shows improved prediction
of the held-out TinyStories text distribution, not demonstrated improvement
in reading comprehension, reasoning, instruction following or general
intelligence.

## Lineage and retained output

- Dataset revision: `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`
- Declared license: `CDLA-Sharing-1.0`
- Parent step: **36,000**, run 42115007553.
- Candidate step: **44,000**, run 42116842094.
- Candidate archive SHA256:
  `ca9792c3c68cd1db915c7dedc60149013ba7d981781783b2f03a1e7ecbce84a1`
- Artifact: `forge-seed-v07-continuation`, retained through
  **2027-02-03**. Its `lineage.json` records exact parent/candidate hashes
  and verifies data reuse and optimizer-state resume.

## Decision

Use the 44,000-step checkpoint as the parent of the next bounded approved
core-language run while validation loss continues to improve. Keep the
checkpoint research-only pending broader language and human-reviewed quality
gates; do not replace Android weights automatically.
