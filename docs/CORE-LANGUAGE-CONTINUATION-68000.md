# Forge Seed v0.7 Core Language Continuation — 68,000 Updates

**Run date:** 2026-11-05  
**Status:** Completed successfully; research checkpoint retained, not promoted  
**Actions run:** [continuation 42121807463](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42121807463)  
**Parent run:** [60,000-step continuation 42120062062](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42120062062)

## Training result

This continued Forge Seed's original scratch-initialized language model from
step 60,000, resuming its AdamW optimizer and RNG states. It completed 8,000
next-token updates using the exact same pinned TinyStories train/validation
split, tokenizer, 2.22M-parameter architecture, seed 43, batch size 4,
context 128 and learning rate 0.0003.

- Total steps: **68,000**
- Additional updates: **8,000**
- Runtime: **602.34 seconds**
- Same-protocol 64-window progress loss: **2.184636 at step 60,000 →
  2.147870 at step 68,000** (1.68% reduction).
- Cumulative 64-window curve change: **2.369578 at step 28,000 →
  2.147870 at step 68,000** (9.36% reduction).
- Separate final 256-window diagnostic: **2.177065**, down from **2.205533**
  at step 60,000 under the same final-evaluation method (1.29% reduction).
- Sampled estimated bits per byte: **0.862625**; estimated from 256 token
  windows, not an exact full-validation measurement.

## Capability check

The bounded forced-choice diagnostic scored **20/40 (50%)** and **0/20**
mirrored pairs had both answer orders correct. It remains at chance on the
small template-based check. The measurable gain is in held-out text
prediction, not established comprehension, reasoning, instruction following
or general intelligence.

## Durable lineage

- Dataset revision: `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`
- Declared license: `CDLA-Sharing-1.0`
- Parent step: **60,000**, run 42120062062.
- Candidate step: **68,000**, run 42121807463.
- Actions artifact SHA256:
  `8603c8789d06ca3257eb4dc690a5985e080599bcd80d0a8ccc8f0d3b64f25963`
- Unpublished durable draft archive:
  [Forge Seed v0.7 — 68,000-step research checkpoint](https://github.com/tpylman90-ctrl/Forge-Core/releases/tag/untagged-eb899a66cc117bd0ee63).
  It includes parent and candidate checkpoints, optimizer state, exact public
  split, source manifest, metrics, capability check and SHA256 lineage.
  The matching Actions artifact expires **2027-02-03**.

## Decision

Use this as the parent for another bounded continuation only while the matched
held-out text signals improve. Keep it research-only and do not replace Android
weights automatically; the capability gate remains unmet.
