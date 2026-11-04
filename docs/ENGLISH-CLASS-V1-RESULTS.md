# Forge Seed English Class v1 — First measured learning run

**Status:** completed one-shot research experiment, NO model promotion.
This study tested a basic educational English-class curriculum
against the original Forge Seed scratch-trained language Transformer.

## Verified source, code and compute

- Successful GitHub Actions run:
  https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41775152111
- Temporary full research artifact ID **13296023838**:
  https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41775152111/artifacts/13296023838
- Artifact expires **December 4, 2026** and contains three unpromoted
  trained models, optimizer states, dev history, and the full report.
- Merged implementation:
  `b262d4bc6181e55312c4aa7e65113816eecebf67`, PR #17.
- Original audited scratch-trained Forge Seed v0.7 Actions lineage
  **41572257363**; original tokenizer and model weights from scratch,
  no downloaded pretrained model, phone data, or personal memories.
- Six exercises: uppercase letters, vowels, joining word parts,
  identifying verbs, identifying acting subjects, and sentence order.
- 384 original synthetic training questions, 72 development questions,
  and 96 different synthetic exam questions using disjoint grammar
  lexicon for names/objects/verbs/word components and question
  phrasings. **A/B positions balanced by skill.**
- **720 optimizer updates per research arm** (120 at each of six
  levels), each with 20% public-source language rehearsal.
  All arms start from identical authenticated original Forge Seed
  weights. Ordered and mixed arms see exactly the same multiset of
  practice batches, only in different order. The corrupted control
  shuffles correct letters within each skill while preserving label
  frequency. All trained weights remain experimental and isolated.
- **153.2 seconds** total experiment training time.
- Passed all synthetic source-answer integrity, anti-leakage,
  matched-schedule and gradient tests.

## Previously unused synthetic exam results

| Method | Exact answers / 96 | Accuracy | Development accuracy /72 |
| --- | ---: | ---: | ---: |
| Frozen original Forge Seed | 48 | 50.00% | Baseline recorded in artifact |
| Ordered six-level English Class | 48 | 50.00% | 50.00% |
| Same examples, mixed throughout | **53** | **55.21%** | **63.89%** |
| Corrupted answers, ordered schedule | 47 | 48.96% | 54.17% |

This is a **small descriptive difference**, not a statistically
established language-learning gain or a preschool proficiency gate.
The staged course did not exceed the original 50% exam baseline.
The mixed model gained five exact answers on 96 questions relative
to the original, and six relative to shuffled teaching. The 96-case
exam is now **research-used**, not an untouched test for a future
curriculum or model.

### Language retention bookkeeping correction

The first training script mistakenly evaluated the **frozen** TinyStories
validation loss over **32** fixed windows and the trained branches
over **48**. Therefore the original reported retention ratios
(0.9998, 1.0119, 1.0051) **are NOT directly comparable**.

PR #18 corrected the implementation and PR #19 implemented an
independent read-only audit without expensive retraining.
**This audit completed successfully and verified the original checkpoint
and tokenizer SHA256 lineage and all saved model predictions.**

- [Successful retention audit run 41775982211](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41775982211)
- [Audited lightweight result artifact 13297475570](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41775982211/artifacts/13297475570)
- Same 48 fixed validation windows and 64-token contexts for frozen
  and each development-selected adapted model
- Source frozen loss **2.607221**; corrected relative losses:

| Method | Adapted loss / frozen loss | Within 5% preliminary retention limit |
| --- | ---: | --- |
| Ordered English Class | **0.99973** | Yes |
| Equal-exposure mixed lessons | **1.01182** | Yes |
| Corrupted label control | **1.00508** | Yes |

These are **matched small-window public-text loss measurements**, not
proof of fluency or memory retention of personal conversations.
Mixed-order lessons increased held-out loss by about **1.18%**
despite their slight multiple-choice exam advantage.

### Subject-by-subject exam counts, independently rescored

Each subject has 16 A/B test questions; **8/16 equals 50%**.
The original frozen model scored exactly 8/16 on every category.

| Subject | Frozen | Ordered course | Mixed course | Corrupted control |
| --- | ---: | ---: | ---: | ---: |
| Capital letter forms | 8 | 8 | 7 | 7 |
| Vowel versus consonant | 8 | 8 | **10** | 8 |
| Joining word parts | 8 | 8 | **4** | 8 |
| Noun versus verb | 8 | 8 | **12** | 8 |
| Sentence subject | 8 | 8 | 8 | 8 |
| Subject–verb–object ordering | 8 | 8 | **12** | 8 |
| **Total** | **48/96** | **48/96** | **53/96** | **47/96** |

The mixed model shows some skill-specific progress, but word-building
**deteriorated** markedly. The ordered model shows no detectable
improvement over frozen baseline on this exam. With only 16 items
per subject, none of these should be overinterpreted.

The exam has already been inspected and rescored as a model-integrity
check, **not** as a second independent test. Future experiments must
use new, separately authored tests rather than tuning to this exam.

## Decision and next course

**Do not promote any experimental language weights.** What is
established is infrastructure: an English-class syllabus with
independently checkable gold rules, controlled order comparisons,
original source-preserving rehearsal, and a real CPU training run.

For the next *training* experiment, evaluate grammar skills that go
beyond A/B recognition—singular/plural agreement, verb tenses,
negation, short sentence completion, and reversible actor/object
roles—using *independently authored* examples. Retain public text
rehearsal. Consider a revised schedule mixing the classes because
it appeared slightly better in this first pilot, but replicate it
on fresh exams and seeds before attributing gains to mixing.

This is deliberately **text-first**. Alphabet/vowel classification
does NOT teach English sound recognition or pronunciation without
appropriate phonetic/audio information.

[Curriculum design and limitations](ENGLISH-CLASS-V1.md).
