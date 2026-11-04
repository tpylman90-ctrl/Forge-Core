# Forge RLVR v7 — Increasing exact formal proof depth

**Verified controlled experiment; no model checkpoint promoted.**

- Successful workflow: https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41770908240
- Temporary research artifact: https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41770908240/artifacts/13295190435
- Research implementation merged in commit
  `b90f0a087b4c8c091e1432c4f29faa152442b2cb` (PR #15).
- Two random initializations, **5321** and **5322**.
- **900 updates per arm per seed**. Three separate arms:
  flat long-proof imitation, staged 1→2→3-step imitation,
  and staged imitation with a small additional RLVR policy term.
- CPU experiment completed in **78.1 seconds**; temporary artifact
  expires **December 4, 2026**.
- Oracle now verifies **shortest** action traces for curriculum labels,
  fixing a CI-detected mistake where a one-step task could have
  been taught using a valid three-step detour.
- Checkpoint selection used development results, not shifted metrics.
- Every shifted test level contains **240** tasks per seed:
  **120 formally solvable** and **120 impossible**.
  Results below count fully verified completed proofs out of
  solvable cases, rather than success at selecting an individual action.
- Original Forge Seed language model, neural pretraining artifacts,
  the independently developed Android app and private memories
  remain entirely unchanged.

## Shifted, exact proof completion

| Arm | Seed 5321 1-step /120 | Seed 5321 2-step /120 | Seed 5321 3-step /120 | Seed 5322 1-step /120 | Seed 5322 2-step /120 | Seed 5322 3-step /120 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flat long-proof imitation | 1 | 0 | 0 | 23 | 16 | **10** |
| Staged imitation | **50** | **4** | 0 | **43** | 15 | 2 |
| Staged + RLVR | 40 | 1 | 0 | 44 | 13 | 2 |

Aggregated across both initializations, out of **240 feasible
tasks at each depth**:

| Arm | One-step | Two-step | Three-step |
| --- | ---: | ---: | ---: |
| Flat | 24/240 (**10.0%**) | 16/240 (6.7%) | **10/240 (4.2%)** |
| Staged | **93/240 (38.8%)** | **19/240 (7.9%)** | 2/240 (0.8%) |
| Staged + small RLVR | 84/240 (35.0%) | 14/240 (5.8%) | 2/240 (0.8%) |

These figures show that the new staged curriculum produces a
large improvement on one-step problems in this small procedural
test family, but **does not improve full three-step proofs**.
The flat curriculum is stronger at the longest evaluation depth.
Extra RLVR after imitation did not improve the stage-wise results.

Performance differs considerably across model initialization seeds.
No formal statistical significance or independent human-designed
generalization claim follows from only two seeds.

## Interpretation

1. The independently validated shortest-proof curriculum fixes the
   previous information bottleneck: the model now learns to propose
   some valid single formal operations without being offered
   precomputed next-state answers.
2. One-step success is not compositionality. The later proof steps
   remain much weaker, particularly on shifted arithmetic values
   and altered logic premises.
3. Staged training devoted too little late compute to genuinely
   three-step goals. The last stage mixed all depths equally, so
   the longest proofs received a smaller share of examples than
   in the flat control. This is a plausible contributor, not a
   verified causal explanation.
4. The architecture is an original randomly initialized, separate
   policy trained with **oracle-generated labels**. This is
   supervised process learning plus optional RLVR, not an
   independently conversational AI or unassisted reasoning.
5. A deterministic exact solver remains the authoritative verifier;
   it is not evidence that the learned neural model has already
   mastered formal operations.

## Research decision and recommended next gate

**NO PROMOTION.** Do not connect experimental proof policies to the
canonical Forge Seed language model or claim preschool-level reasoning.

Future controlled work should maintain a minimum fraction of
three-step examples throughout late training while retaining
short proofs as rehearsal. Compare equal-depth exposure and
initialization seeds and use truly new operation families before
asserting transfer. We also need a planning model that can
compose actions using externally verified intermediate outcomes
and report precise verifier calls, invalid proposal rates and
full-proof safety.

This outcome provides a concrete learning signal: **the first
formal action is more learnable than a complete multi-step proof**.
Preserve this finding and do not spend another long GitHub Actions
session until the next curriculum/test design addresses the
depth-transfer failure.

[Experimental method and isolation rules](RLVR-DEPTH-CURRICULUM-V7.md).
