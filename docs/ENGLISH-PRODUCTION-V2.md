# Forge English Class v2 — learning to write words and sentences

## Why this follows the verified v1 experiment

The first six-stage English Class adaptation taught capital letters,
vowels, word parts, nouns/verbs and simple sentence structure to the
existing scratch-trained Forge Seed. In the [audited v1 results](ENGLISH-CLASS-V1-RESULTS.md),
the frozen model and ordered class each scored **48/96** two-option
questions; mixed-order teaching scored **53/96**, corrupted-label
teaching scored **47/96**. The improvement was small and uneven:
mixed lessons scored **4/16** on combining word parts. A source-audited
48-window TinyStories check measured only a 1.182% increase in
next-token loss. No weights were promoted.

The key weakness is that A/B recognition is not language production.
V2 requires the original language network to **generate exact short
answers without multiple-choice options**.

## Lesson progression

Seven original English-teaching categories with deterministic
independent grammar oracles:

| Skill | Example task | Verified written output |
| --- | --- | --- |
| Join written parts | `parts=sun+shine` | `sunshine` |
| Plural nouns | `noun=daisy` | `daisies` |
| Present-tense subject agreement | `verb=carry` | `carries` |
| Regular past tense | `verb=try` | `tried` |
| Actor/receiver roles | `event=Amina\|give\|kite\|Benton;ask=receiver` | `Benton` |
| Construct a full sentence | `sentence=Amina\|carry\|kite` | `Amina carries the kite.` |
| Negate a full sentence | `sentence=Amina\|carry\|kite` | `Amina does not carry the kite.` |

All labels are derived *again* by an independent explicit grammar
parser. For actor roles, reversing participants reverses the
correct answer. For spelling, the rule checker handles elementary
regular **-s, -es, -ies, -ed, -d and -ied** transformations. These
rules are deliberately limited: irregular verbs, phonics, complex
clauses and conversational language are later milestones.

The neural model receives a source expression and a written question,
**not the gold answer or A/B options**. Its causal LM objective
predicts only target answer tokens; public TinyStories context is
rehearsed in each training step. Evaluation uses deterministic greedy
text generation with exact normalized written-answer matching;
responses like an unrelated word or no answer score incorrect.

## Data and test separation

The 448 training lessons per seed use vocabulary and names disjoint
from 56 development and 112 final exam lessons; the development and
exam sets are also lexically disjoint. Question phrasings vary
between splits. Source templates and abstract grammar rules are
still shared. This is a synthetic generalization test, **not**
human-authored English literacy certification.

The v1 final exam is not reused. The v2 exam is specified in source
before training and used **once per saved development-selected
policy**; we must mark it research-used after analysis.

## Equal-budget arms and safeguards

From the **same original Forge Seed v0.7 scratch checkpoint**
(Actions source run 41572257363), compare:

1. `verified_mixed`: verified grammar demonstrations interleaved
   across all seven skills;
2. `verified_staged`: exactly the same training batch multiset,
   ordered by skill with earlier skill rehearsal;
3. `shuffled_mixed`: same prompts and schedule, but within-skill
   target answers shuffled to disallow true grammar learning;
4. `text_only`: same number of optimizer updates using only
   licensed public TinyStories rehearsal, without grammar labels.

All learning branches use same architecture, train update count,
model starting weights, source tokenization and public text data.
The first three use a **75% generated-answer learning objective**
plus **25%** public-text rehearsal. The fourth uses only
public-text loss and is therefore a different objective but matched
for optimizer step count, not teacher information.

Run two different training-data/schedule seeds, **8123 and 8124**,
with **80 updates per each of seven subject blocks**, **560 updates
per arm per seed**, all sequentially on one CPU-hosted GitHub Actions
runner with a 45-minute job ceiling. No recurring automatic tests.

Checkpoint selection uses only the independently generated
development set, prioritizing exact open-ended answers and the weakest
skill. The final 112-problem synthetic exam and fixed-window public
validation loss are measured after selection. Score each skill
individually, show short samples of real generated answers and
compare to the unchanged frozen baseline.

**Language retention check:** exact same 48 fixed upstream held-out
TinyStories validation windows of 64 BPE tokens for frozen and all
adaptations. A ratio over 1.05 fails the preliminary preservation
threshold; below that does not imply true fluency.

## Boundaries and gates

This experiment changes only files under the separate neural
research code, its tests, documentation and dedicated workflow.
It does not alter `android/`, mobile APK signing, personal memory,
RLVR proof policies or the canonical Forge Seed checkpoint.

No external pretrained models or private examples are used.
Temporary experiments include weights and optimizer/RNG metadata
for audit; they expire under GitHub's artifact retention policy.

Success requires a **repeatable meaningful improvement above frozen,
shuffled and text-only controls**, real word/sentence exact matches
on unfamiliar examples, and acceptable held-out text preservation
across both seeds. A 0%–5% exact score is still a failed
open-ended language-learning experiment, no matter the training loss.
Any improvements would require externally authored grammar tests,
independent seeds and human-reviewed sample quality before neural
weight promotion or integration into Android.

This tests elementary written morphology and sentence production,
**not sound pronunciation or listening**, and not general or
human-level reasoning.
