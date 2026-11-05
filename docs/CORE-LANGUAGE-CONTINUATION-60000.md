# Forge Seed v0.7 Core Language Continuation — 60,000 Updates

**Run date:** 2026-11-05  
**Status:** Completed successfully; research checkpoint retained, not promoted  
**Actions run:** [continuation 42120062062](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42120062062)  
**Parent run:** [52,000-step continuation 42118418883](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42118418883)

## Training result

This continued the original scratch-initialized Forge Seed language model from
step 52,000, resuming its AdamW optimizer and RNG states. It performed 8,000
next-token updates on the exact same pinned TinyStories training and validation
split, retaining the tokenizer, 2.22M-parameter architecture, seed 43,
batch size 4, context 128 and learning rate 0.0003.

- Total steps: **60,000**
- Additional updates: **8,000**
- Runtime: **523.96 seconds**
- Same-protocol 64-window progress loss: **2.201201 at step 52,000 →
  2.184636 at step 60,000** (0.75% reduction).
- Separate 256-window final diagnostic: **2.205533**, down from **2.226931**
  at step 52,000 under the same 256-window final-evaluation method.
- Sampled estimated bits per byte: **0.873905**, estimated from 256 token
  windows, not an exact full-validation measurement.

## Capability check

The bounded forced-choice diagnostic scored **20/40 (50%)** and **0/20**
mirrored pairs had both answer orders correct. It remains at chance on this
small template-based check. The result supports only a narrow language-model
claim; it does not demonstrate better reading comprehension, reasoning,
instruction following or general intelligence.

## Lineage and retained output

- Dataset revision: `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`
- Declared license: `CDLA-Sharing-1.0`
- Parent step: **52,000**, run 42118418883.
- Candidate step: **60,000**, run 42120062062.
- Candidate Actions artifact SHA256:
  `aeaa54b5e9b90d019014d2512e2ff830b2e41933e13dbc1a8bd73ded16cda0dd`
- Candidate artifact expires **2027-02-03**. It includes the parent and
  candidate checkpoints, exact public split, metrics, capability check and
  checkpoint hash lineage. The durable unpublished draft archive is being
  created separately.

## Decision

Continue bounded next-token learning while the same held-out signals improve.
Keep this checkpoint research-only until broader language-quality and
independent capability gates pass. Do not update Android weights automatically.
