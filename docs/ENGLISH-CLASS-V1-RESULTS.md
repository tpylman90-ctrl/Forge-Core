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

A separate PR #18 fixed the source code to use 48 windows for both,
and a follow-up *read-only* study, PR #19, audits all three archived
models without spending time on duplicate training. **Until the
read-only audit is verified, do not infer language was retained or
regressed from the old ratios.**

The point of the audit is to establish a reproducible common
validation geometry, verify checkpoint/tokenizer lineage and show
per-skill English Class results. No exam is being relabeled as
previously untouched.

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
