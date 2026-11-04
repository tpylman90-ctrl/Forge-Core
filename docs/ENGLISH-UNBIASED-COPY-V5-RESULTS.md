# Forge English v5 — Unbiased counterfactual copying experiment

**Completed and verified November 4, 2026. No experimental weights promoted.**

- [GitHub Actions main v5 run 41886375209](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41886375209) — successful; all CI and artifact steps passed.
- [Temporary research artifact 13353145236](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41886375209/artifacts/13353145236) — contains the original results, development curves, best/last weights and optimizer states for 8 research policy copies.
- [Read-only independent word/nonword transfer run 41887963917](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41887963917) — successful, separately scheduled behind main run.
- [Temporary transfer artifact 13350749842](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41887963917/artifacts/13350749842) — records all inspected transfer metrics.
- [Predeclared v5 methods](ENGLISH-UNBIASED-COPY-V5.md) and [word-transfer methods](ENGLISH-V5-WORD-CHALLENGE.md).

## Study setup and provenance

All arms adapt from independently cloned original **scratch-trained Forge Seed v0.7** language weights and original BPE tokenizer, cryptographically verified against run **41572257363**. No pretrained weights, private user data or Android files. No canonical checkpoint overwritten. Both seeded branches used **900 updates per arm** (2 seeds × 4 arms = 7,200 total updates), 4 source question examples/update, 20% original TinyStories rehearsal, on one GitHub CPU runner. Training and evaluation completed in **642.55 seconds**, with one successful one-off workflow. Checkpoint selection used development-only metrics. Every source is uniformly random 4–8 lowercase letters, with a one-character-altered twin; unlike v4 there is no vowel/consonant parity shortcut.

The final v5 exam has **40 counterfactual families**, two whole-string variants each, **508 individually generated letters per trained seed**. Both seeds are evaluated on the same final exam; combined totals are model/question trials, not 1,016 independently authored tests.

## Unbiased synthetic exam results

| Training condition | Seed 11631 correct /508 | Seed 11632 correct /508 | Combined /1016 | Both-twin correct at changed position /80 | Externally assembled complete strings /160 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Verified counterfactual examples paired within batches | **86** | **87** | **173/1016 (17.0%)** | **3/80** | **0/160** |
| Identical verified examples, batch order shuffled | **87** | **56** | 143/1016 (14.1%) | 2/80 | 0/160 |
| Source-index labels permuted | 29 | 22 | 51/1016 (5.0%) | 0/80 | 0/160 |
| Source text masked while preserving the target labels | 20 | 27 | 47/1016 (4.6%) | 0/80 | 0/160 |
| Frozen original, same exam | 0 | repeated source | 0 on this exam | 0 | 0 |

The fact that verified teaching exceeds shuffled/hidden-source controls on **both** training seeds supports some source-sensitive signal. However, complete counterfactual pair success is **3/80**, and even externally assembling each separately predicted letter does not produce a correct full string.

### Original public-story language retention

Loss ratios (development-selected adapted model vs identical frozen upstream validation windows):

- Paired verified: **1.02490**, **1.02893**.
- Same examples unpaired: **1.02197**, **1.02057**.
- Permuted labels: **1.01149**, **1.00965**.
- Masked source: **1.00374**, **1.00860**.

All are within the project's **preliminary 5% held-out next-token loss regression gate**, but that is not evidence of retained fluent conversation. No model was promoted.

## Read-only familiar English-word / edited-nonword transfer

The transfer exam was preregistered in source code **before the final v5 metrics were inspected**. It uses **36 familiar 4–8-character English words**, each with a **one-character-edited nonword twin**. It queries all positions, **428 letter questions per trained seed**, 72 externally assembled strings per seed and 36 hard changed-letter pairs per seed. The word list is assistant-curated, not independently human authored, and the intact words may have appeared in original public TinyStories pretraining.

| Condition | Seed 11631 correct /428 | Seed 11632 correct /428 | Combined /856 | Both hard twins correct /72 | Full strings externally assembled /144 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Paired verified | **88** | **78** | **166/856 (19.4%)** | **3/72** | 0 |
| Same examples unpaired | 83 | 54 | 137/856 (16.0%) | 2/72 | 0 |
| Permuted labels | 11 | 2 | 13/856 (1.5%) | 0 | 0 |
| Source masked | 3 | 29 | 32/856 (3.7%) | 0 | 0 |
| Frozen original | 0 | repeated source | 0 | 0 | 0 |

These results support **modest transfer of the indexing behavior to recognizable word sources**, relative to the controls, but **do not establish meaningful word spelling, sentence understanding or independent language production**. Modified twins are nonwords; external assembly is a deterministic loop controlled by the tester rather than by Forge.

## Decision and next gated experiment

**NO PROMOTION.** The prior v4 43% letter result was strongly inflated by alternating consonant/vowel positional regularities. v5's 17% result is harder to fake and beats appropriate controls, but only **3/80** changed-character pairs were both correct and **zero** complete novel sequences were assembled correctly. Further repetition of the same training objective is unlikely to build robust, independent word copying without architectural work.

Recommended bounded next research: a trainable, originally initialized **character-aligned pointer/copy component** (no pretrained model), compared with the original BPE Transformer and a nonlearned exact-copy oracle. Use a new held-out source family with unstructured letters, novel lengths, and changed-letter twins. Distinguish performance of the new component from the original Forge language model. Preserve exact answer/oracle validation and original public-story loss; only then consider a separate bridge for independently generated entire words.

The existing research proposal [English source-copy architecture options](ENGLISH-COPY-ARCHITECTURE-OPTIONS.md) gives candidate designs, limitations and references. Continue Android application/signing in its other isolated conversation.
