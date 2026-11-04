# Forge English v4 — explicit source-position letter learning results

**November 4, 2026. Verified hosted CPU study; experimental weights NOT promoted.**

[Successful GitHub run 41853797859](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41853797859) · [Research artifact 13335154414](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41853797859/artifacts/13335154414), expires 2026-12-04. Implementation merged as PR #23, commit `eb0ea0af30cbc26d4aab127dcb5644259752ab00`.

All branches started from the independently audited original scratch-trained Forge Seed v0.7 Transformer, with no pretrained weights or Android data. Two training/schedule seeds 10331 and 10332; five arms; 440 optimizer updates per arm and seed; 384.34 s training/evaluation. The exact original Forge Seed model and tokenizer remain untouched.

The independent synthetic final exam contains **48 distinct 3–7-letter source strings, each queried at every position**, giving 249 letter queries per independently trained model. Thus, combining the two seeds produces **498 model-query trials** on the **same exam set**, not 498 independently authored questions. External assembly tests 48 strings per model (96 model–string trials). Training/dev/exam string sets were separated by a new SHA256 namespace, and final question wording differed. The exam has now been used and cannot be recycled as a sealed validation set.

## Verified generated-letter results

| Condition | Seed 10331 correct letters /249 | Seed 10332 /249 | Combined /498 | Later positions (2–7) /402 | Externally assembled complete strings /96 | Frozen public-story loss ratio, seeds 10331 / 10332 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Verified mixed lessons | **109** | **105** | **214/498 = 43.0%** | **173/402 = 43.0%** | **3/96 = 3.1%** | **1.01216 / 1.01161** |
| Verified staged lessons | 57 | 71 | 128/498 = 25.7% | 110/402 = 27.4% | 1/96 = 1.0% | 1.02183 / 1.01679 |
| Labels shuffled within position | 35 | 32 | 67/498 = 13.5% | 56/402 = 13.9% | 0/96 | 1.00854 / 1.00198 |
| Train on first position only | 54 | 54 | 108/498 = 21.7% | 12/402 = 3.0% | 0/96 | 1.00949 / 1.00228 |
| Original public stories only | 0 | 0 | 0/498 = 0% | 0/402 | 0/96 | 0.99907 / 0.99351 |
| Untouched source model | 0 | same model | 0 in source exam | not applicable | 0 | 1.0 |

The two mixed-training seeds agreed reasonably closely on total letter accuracy (109/249 vs 105/249). The externally assembled string successes were 2/48 and 1/48. The exact symbolic assembly code merely concatenates separately requested neural letters. **It is NOT evidence that Forge autonomously wrote a word.**

### Position-specific transfer for the mixed model, combined across seeds

| Explicit source index | Exact letter answers |
| --- | ---: |
| 1 | 41/96 = 42.7% |
| 2 | **84/96 = 87.5%** |
| 3 | 31/96 = 32.3% |
| 4 | 21/78 = 26.9% |
| 5 | 18/62 = 29.0% |
| 6 | 15/46 = 32.6% |
| 7 | 4/24 = 16.7% |

**Caveat:** This curriculum synthesizes strings using alternating consonant and vowel positions, so second-position questions are systematically easier (always vowels). The high 87.5% at index 2 must not be presented as normal arbitrary-position language ability. On positions **3–7 only**, mixed training scored **89/306 = 29.1%**, against **37/306 = 12.1%** for shuffled labels. That is still an improvement on this restricted synthetic task, but much weaker than index 2.

Some generated responses to a *single-letter* prompt contained multiple letters or familiar learned fragments, breaking assembly. The study does not establish open-ended spelling, learned iteration, independent word termination, grammatical comprehension, or conversation.

## Decision

**Do not promote.** There is a defensible, repeatable **learning signal in individual character retrieval** relative to the chosen controls, and original story loss stayed within the exploratory 5% regression limit. That does NOT constitute robust language proficiency.

Before a further long run, improve the curriculum/test design:
- Hold out novel strings with **unrestricted position distributions**, including adjacent consonants and vowels, so a model cannot infer the letter from artificial position parity.
- Evaluate letter accuracy separately for each source position and for **hard negatives**, e.g. two strings differing only in one letter at the queried position.
- Add an **independent symbolic reference** to distinguish exact copy functionality from learned text inference; preserve shuffled-letter, first-only and story-only baselines.
- Evaluate whether the original model can emit strictly one letter and a delimiter, not multiword filler.
- Try **masked-character reconstruction** or an original lightweight pointer-like encoder/decoder as a controlled architecture change. The model weights remain scratch-created and any proposed new mechanism must be audited independently.
- Resume complete word and sentence production only when novel multi-position character copying demonstrably generalizes outside the alternating consonant-vowel synthetic generator.
- Retain validated public-story loss and independently authored fresh exams before any neural checkpoint promotion.

[Research design, controls, source isolation](ENGLISH-POSITION-COPY-V4.md) · [Prior source-character experiment](ENGLISH-CHAR-COPY-V3-RESULTS.md).
