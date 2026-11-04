# Forge Seed v2 — verified 36-minute paired-question experiment

**Completed November 3, 2026; research-only, no promotion.**

- Successful [Actions run 41728138301](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41728138301)
- [Download temporary source artifact](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41728138301/artifacts/13278198241)
- Artifact ID **13278198241**, created Nov 3, expires **Dec 3, 2026**
- Synthetic training **320** questions, development **80**, fresh final test **360** per seed
- Audited scratch-model ancestry; original language backbone frozen
- Three independent seeds **7382, 7383, 7384**
- Separate paired-contrastive, cross-entropy, question-ablated and shuffled-group conditions
- Training **2157.96 seconds**, approximately **35 min 58 s** across all seeds
- **28,382 + 28,198 + 27,641 = 84,221** optimization rounds per arm across seeds
- Mean throughput **39.118 rounds/second** across all arms (each round updated 4 heads)
- Each new head **654,721** parameters; peak Linux process RSS approximately **530 MiB** as reported
- Evaluation selection uses development results, never the final test

## Held-out results

Three seeds times 360 questions = **1080** individual answers.
Three seeds times 180 event pairs = **540** paired giver/recipient tests.

| Arm | Exact individual answers | Both role questions correct | Both entity swaps correct | Answers changed when role question changed |
| --- | ---: | ---: | ---: | ---: |
| Paired contrastive | **597/1080 (55.3%)** | **95/540 (17.6%)** | 230/540 | 71/270 |
| Cross-entropy-only | 581/1080 (53.8%) | 81/540 (15.0%) | 229/540 | 70/270 |
| Question-ablated | 540/1080 (50.0%) | **0/540 (0.0%)** | 262/540 | 0/270 |
| Group-flipped-label control | 517/1080 (47.9%) | 63/540 (11.7%) | 158/540 | 79/270 |

Paired scores by seed:
- 7382: **40/180** contrastive; 34/180 cross-entropy; 0/180 question-ablated; 11/180 shuffled
- 7383: **28/180** contrastive; 22/180 cross-entropy; 0/180 question-ablated; 18/180 shuffled
- 7384: **27/180** contrastive; 25/180 cross-entropy; 0/180 question-ablated; **34/180 shuffled**

### Critical interpretation

Some benefit from question-conditioned contrastive learning appears, but
the overall paired result **17.6%** is still below the nominal **25%**
for two independent uniformly random candidate choices. Predictions are
correlated, so the independent-choice reference is not a directly
interchangeable baseline. Most of the time Forge did not change its
answer when asked the opposite role; the question-ablated arm's 0%
paired score confirms that the question signal is necessary in this
architecture, not that semantics are mastered.

One shuffled-label control exceeded the verified arm on an individual
seed. This is a substantial warning against claiming stable
generalization.

Development-selected best checkpoints were at varied steps:
- Paired: **9000, 22500, 16500**
- Cross-entropy: **3000, 2000, 10500**
- Shuffled: **26000, 24500, 1500**

Extending all runs to the full 36 minutes without development-based
selection would not be justified. The original frozen Forge Seed model
has not been replaced.

### Next experiment

[Factorized v3 role-question research](ROLE-FACTORIZED-V3.md) tests
separate story-giver learning and question-intent classification,
compares a corrected group-corrupted null, and uses a new,
previously unseen synthetic exam. The v2 final test is now
research-used, **not** untouched for future evaluation.

This result establishes CPU throughput and experiment feasibility,
**not a daily entitlement to GitHub Actions compute**, and it is not
evidence of general natural-language understanding.
