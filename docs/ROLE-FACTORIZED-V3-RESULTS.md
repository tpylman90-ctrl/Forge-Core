# Forge Seed v3 — verified factorized question/event research

**Research-only. No neural head promotion.**
**November 3, 2026 (local time); Actions run ended November 4 UTC.**

## Provenance and execution

- [Successful GitHub Actions run 41743517853](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41743517853)
- [Research artifact 13281429407](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41743517853/artifacts/13281429407), expires December 4, 2026
- Original audited scratch-trained Forge Seed v0.7 checkpoint/lineage restored read-only; all its parameters **frozen**
- Independent seeds **8311, 8312**; **1800 updates per seed per training arm**
- Training questions **1600**, development questions **80**, new synthetic final questions **288 per seed**
- Four arms: factorized verified event/question; factorized randomly group-corrupted events; factorized question-ablated; historic monolithic question-conditioned head retrained on identical curriculum
- Head parameters: factorized **507,075**, monolithic **654,721**
- **85.53 seconds** total experiment time; approximately **42 seconds per seed**
- Source-only entity detector; independent gold parser for final test
- Checkpoint selection by development data, new synthetic exam scored after training
- All weights and optimizer/RNG recovery states saved as research-only artifacts

## Held-out paired-role results

Per seed, **144** separate event/question pairs: both giver and recipient
must be correct for an event to count.

| Model arm | Seed 8311 | Seed 8312 | Combined both-correct |
| --- | ---: | ---: | ---: |
| Factorized verified | 33/144 | 10/144 | **43/288 (14.9%)** |
| Factorized wrong-event control | 22/144 | 25/144 | **47/288 (16.3%)** |
| Factorized question-ablated | 0/144 | 0/144 | **0/288 (0.0%)** |
| v2-style monolithic, same training curriculum | 25/144 | 31/144 | **56/288 (19.4%)** |

Correct individual answers out of 576 total across seeds:
- Factorized verified **306/576 (53.1%)**
- Wrong-story control **274/576 (47.6%)**
- Question-ablated **288/576 (50.0%)**
- Monolithic **330/576 (57.3%)**

The paired result is below a nominal 25% reference for *independent*
uniform two-person guesses. This reference is not fully matched to
correlated predictions; nonetheless the head has not learned
reliable opposing answers. The wrong-story arm performed better
than the verified factorized head in the combined paired metric.

## Diagnostic decomposition: the key finding

The factorized architecture separates (a) which name is the **giver**
in the source story and (b) whether a question asks for the **giver**
or **recipient**. No gold labels enter inference.

| Factorized ability on new synthetic exam | Seed 8311 | Seed 8312 |
| --- | ---: | ---: |
| Story giver selection | **204/288 (70.8%)** | **246/288 (85.4%)** |
| Question role classification | **194/288 (67.4%)** | **146/288 (50.7%)** |
| Composed final individual answer | 160/288 (55.6%) | 146/288 (50.7%) |
| Both role questions correct per event | 33/144 (22.9%) | 10/144 (6.9%) |

The event-role learner shows above-50% story-label accuracy on both
seeds. **Question-intent generalization is much less stable**, especially
across novel paraphrases. This makes the question-text understanding
branch a plausible bottleneck, although two seeds cannot establish
reliable population effects.

## Decision

**Do not promote v3.** It is more informative than successful:

1. The original scratch-trained language checkpoint remains unchanged.
2. A factorized learner can sometimes identify which of two people
   gave the item, but answering a novel opposing question remains weak.
3. Synthetic development performance can be misleading; a model with
   strong development paired accuracy can fail on novel final phrasing.
4. More compute on the same 10 training question strings or the same
   synthetic 288-case exam is not justified.
5. The final v3 exam is now research-used and cannot become the
   *untouched* test of a future update.

## Next engineering gate

Prioritize **question-intent language generalization** without freezing
in the current tiny list of manually specified paraphrases:

- Build a transparent question-only evaluation with independently
  authored sentences, withheld paraphrases and balanced giver/recipient
  roles. Include neutral "who" questions as negative examples where
  two-person role assumptions should *not* apply.
- Pretrain a tiny scratch-started question encoder on diverse
  appropriately licensed and reviewed educational question pairs
  (not a pretrained language teacher's weights).
- Compare plain supervised question-role classification with
  contrastively paired giver/recipient paraphrases under matched
  budgets and multiple seeds.
- Only integrate that encoder into event-role predictions if it
  generalizes to unseen wording and whole-story role pairing.
- Preserve all old frozen-language benchmarks and use new sealed
  tests: do not optimize against the already-inspected v3 exam.
- Keep personal why-learning memory separate, private and confirmed.

**Training time is not the objective; held-out comprehension is.**
