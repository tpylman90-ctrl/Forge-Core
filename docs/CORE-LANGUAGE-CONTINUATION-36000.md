# Forge Seed v0.7 Core Language Continuation — 36,000 Updates

**Run date:** 2026-11-05  
**Status:** Completed successfully; research checkpoint retained, not promoted  
**Actions run:** [continuation 42115007553](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42115007553)  
**Parent run:** [scratch language run 42085634121](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42085634121)

## What actually learned

This was a true continuation of Forge Seed's original scratch-initialized
language model. It resumed model weights, AdamW optimizer state and RNG state
from step 28,000 and completed 8,000 more next-token updates. The same
revision-pinned training and validation files, BPE tokenizer configuration,
2.22M-parameter architecture, seed 43, batch size 4, context 128 and learning
rate 0.0003 were retained. No teacher model or private/user data was used.

- Total optimizer steps: **36,000**
- Training updates in this continuation: **8,000**
- Runtime: **441.69 seconds**
- Held-out TinyStories token loss at continuation start: **2.369578**
- Held-out TinyStories token loss after continuation: **2.298014**
- Relative loss reduction during this continuation: **3.02%**
- Sampled estimated bits per byte: **0.910549**; this is an estimate from
  sampled token windows, not an exact full-validation measurement.

## Capability check and interpretation

The separate forced-choice diagnostic scored **20/40 (50%)** after training.
Across 20 mirrored pairs, **0/20** pairs had both answer orders correct. This
is chance-level evidence on that small template-based check. The run therefore
supports a narrow claim: Forge learned to predict the held-out TinyStories
distribution better under this setup. It does **not** show improved reading
comprehension, reasoning, instruction following or general intelligence, and
does not satisfy capability promotion gates.

## Data and checkpoint lineage

- Dataset: pinned TinyStories revision
  `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`
- Declared source license: `CDLA-Sharing-1.0`
- Exact parent split, manifest and checkpoint hashes were verified before
  continuation; the same split was reused for validation.
- Parent checkpoint: step 28,000, seed 43, from run 42085634121.
- Candidate checkpoint: step 36,000, seed 43, from run 42115007553.
- Candidate archive SHA256:
  `1e594220ebc35041372f93258f3217477fb9dd0613ca9da55c90c40fec22c902`
- Candidate archive retained as `forge-seed-v07-continuation` through
  **2027-02-03**. Its `lineage.json` records the exact parent and candidate
  checkpoint SHA256 values.

## Decision

The 36,000-step checkpoint is the next research parent for approved core
language training. It is not installed in the Android app and must not be
described as a generally more intelligent Forge based on this loss change
alone. Continue measured language training while independently researching
more data-efficient learning methods; compare validation loss, fixed samples
and held-out task performance each time. Stop or revise the schedule if the
held-out language metrics regress.
