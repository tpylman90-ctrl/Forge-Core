# Forge Core mistake-curriculum v1 results

**Actions run:** [42133325924](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42133325924)  
**Parent:** Forge Seed v0.7, 76,000 updates, checkpoint from run 42123730003  
**Status:** completed successfully; all candidates remain research-only  
**Durable archive:** [unpublished draft release](https://github.com/tpylman90-ctrl/Forge-Core/releases/tag/untagged-9bdbed374d1df0f60631)

## Design

Nine candidates received 3,000 updates each from identical 76k model weights,
optimizer state, tokenizer, and approved TinyStories training split. The three
arms used sampler seeds 11, 43, and 73:

1. Uniform random replay.
2. A 50/50 mix of uniform sampling and a fixed pool of the highest-loss 20%
   among 2,048 sampled training windows.
3. The same replay plus source-verified correction traces on 10% of updates.
   A trace retained Forge's wrong token, gave the correct token first, and
   stated what the approved training passage actually continued with. This was
   a deterministic source oracle, not an external AI teacher or a semantic
   explanation.

Each candidate used the same 256-window validation protocol as the parent.
Every candidate also received the existing mirrored forced-choice check.

## Results

Validation loss is mean ± sample SD across three sampler seeds; lower is
better. Parent loss was 2.158828.

| Arm | 256-window validation loss | Change vs parent | Forced choice |
|---|---:|---:|---:|
| Uniform | 2.156569 ± 0.002048 | -0.002259 (-0.10%); all 3 seeds improved | 58/120 (48.3%) |
| Surprisal replay | 2.477177 ± 0.011446 | +0.318349 (+14.75%); all 3 regressed | 60/120 (50.0%) |
| Source-verified traces | 2.493345 ± 0.012218 | +0.334517 (+15.50%); all 3 regressed | 59/120 (49.2%) |

The forced-choice result remains at chance. Correct mirrored pairs were 0/60
for uniform, 0/60 for surprisal replay, and 2/60 for source-verified traces.
This does not show a reasoning or comprehension gain.

Individual 256-window losses:

| Arm | Seed 11 | Seed 43 | Seed 73 |
|---|---:|---:|---:|
| Uniform | 2.156760 | 2.158514 | 2.154432 |
| Surprisal replay | 2.489441 | 2.475312 | 2.466777 |
| Source-verified traces | 2.502750 | 2.479535 | 2.497749 |

## Interpretation and decision

Uniform continuation produced a small, consistent next-token validation
improvement. The best of those candidates is seed 73 at 2.154432; it remains a
research checkpoint and was not installed in Android.

The hard-example method concentrated half of ordinary batches on a fixed,
small high-loss pool. The correction arm also repeated a narrow set of traces.
All six candidates using these concentrated schedules lost substantially on
the full held-out text distribution. The evidence is consistent with
overfitting to the narrow pools; it does not support using this replay strength
or this correction-trace mix for core-model continuation.

Keep the uniform seed-73 candidate as the next language-training parent.
Retest uncertainty sampling only with a much smaller, refreshed share of
examples and a stronger ordinary-language replay anchor. Any future teacher
experiment should include genuinely informative, independently verified
explanations; this study did not run an external AI teacher.

No candidate was promoted to the Android app. The complete checkpoint
candidates, metrics, capability outputs, source split, and lineage are in the
draft archive and run artifact.
