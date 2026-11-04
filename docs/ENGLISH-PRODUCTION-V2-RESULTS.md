# Forge English Class v2 — exact written-word and sentence production

**Research-only NEGATIVE RESULT. NO WEIGHTS PROMOTED.**

- [Successful source-verified GitHub CPU workflow 41834951922](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41834951922)
- [Temporary results, original grammar predictions and experimental checkpoints (artifact 13327065217)](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41834951922/artifacts/13327065217)
- Merged implementation: `6806594ae945876cd14ad1414fcdc8203fead17c` (PR #20)
- Original scratch-trained Forge Seed v0.7, audited source run **41572257363**, including exact original tokenizer and publicly sourced TinyStories checkpoint. Never overwritten.
- Two training/schedule seeds **8123 and 8124**.
- Four arms each seed: verified mixed lessons, verified staged lessons, wrong-answer shuffled lessons, and equal-update public-text-only control.
- **560 optimizer updates per arm per seed**, 7 generative grammar skills, 448 training lessons/seed, 56 development exercises, 112 synthetic final exam items. Grammar arms use 25% public-story rehearsal.
- **409.59 seconds** elapsed training/examination time.
- Question phrasings and person, noun, verb and compound-word vocabulary differ across train/development/exam. Abstract grammar families and task formatting remain shared; the exam is now **used**.
- Source labels independently reconstructed from written regular spelling, tense, sentence ordering, subject and semantic-role rules; CI validated dataset, gradients and controls.
- Every final result is from greedy *generated text*, not choosing between A/B options or evaluating teacher-forced likelihood.

## Accurate final scores — held-out open-ended answers

| Arm | Seed 8123 /112 | Seed 8124 /112 | Combined /224 |
| --- | ---: | ---: | ---: |
| Frozen original Forge Seed | 0/112 | 0/112 | 0/224 (same model/exam repeated, not independent trials) |
| Correct labels, mixed lessons | **0** | **0** | **0/224** |
| Correct labels, staged lessons | **0** | **0** | **0/224** |
| Shuffled answers, mixed lessons | **0** | **0** | **0/224** |
| TinyStories-only training | **0** | **0** | **0/224** |

**All seven categories were 0/16 per arm and seed on the final
synthetic exam:** joining written parts, noun plurals, present-tense
verb agreement, regular past tense, identifying giver/recipient,
writing positive complete sentences, and writing negative sentences.
Every development checkpoint also scored 0/56 on exact generated
answers, including the final update for every arm.

The saved model weights were selected using development scores only.
All tied at zero; the implementation chose the latest checkpoint for
each arm. That is NOT an evidence-based best-performing checkpoint.

## Measured training progress and actual output samples

Teacher-forced lesson loss in one correct-label mixed arm dropped
from **9.08248** (first step) to **0.13486** (560th step).
This does **not** imply generalized English production.

From the archived generation previews:

| Unseen source / target | Correct answer | Generated from correct-label training |
| --- | --- | --- |
| Compound with new parts | `mapleflower` | `cherries` (seed 8123 mixed) |
| Plural noun `pillow` | `pillows` | `windows` (seed 8123 mixed) |
| Present verb `follow` | `follows` | `firewalk` (seed 8123 mixed) |
| Giver/recipient event | `Pippa` | `Davin` (seed 8123 mixed) |
| New Soren/box sentence | `Soren finishes the box.` | `Flora does not move the window.` (seed 8123 mixed) |
| New negated river event | `Soren does not visit the river.` | `Flora does not move the window.` (seed 8123 mixed) |

The frozen model tended to emit generic TinyStories-style quoted
dialogue. The adapted models learned to emit classroom-answer-shaped
responses, often terminating with a newline, but largely repeated
**known training vocabulary and entire familiar sentences** rather
than copying supplied unfamiliar entities and applying morphology.

That is consistent with **memorization and weak variable binding**,
not correct generalization. It also means the zero exact score is
not solely a failure to output a newline; some generations were
well-formed training-style answers with the wrong content.

## Measured held-out language preservation

A consistent 48-window, 64-token original TinyStories validation
loss was computed for the frozen and every adapted checkpoint.

| Arm | Seed 8123 / frozen loss | Seed 8124 / frozen loss |
| --- | ---: | ---: |
| Verified mixed | 1.01676 | 1.01606 |
| Verified staged | 1.02402 | 1.02975 |
| Shuffled labels | 1.01105 | 1.01281 |
| Public-text only | 0.98482 | 0.99290 |

All grammar arms are within the preliminary 5% retention-loss gate,
but did not learn novel generated answers. The public-only control
improved held-out next-token loss on this fixed small sample.

## Engineering interpretation and next gate

**Do not increase training time on this identical curriculum.**
The model must learn to represent and **copy unfamiliar variable
content**, before we judge its ability to apply spelling or sentence
transformations to new people and words. The post-training samples
show entire memorized answers in place of new subjects/verbs.

Recommended separate, controlled *letter and word operations* pilot:

1. First teach exact copying of individual lowercase letters, then
   2–5-character letter sequences (including never-seen combinations).
2. Show a word split explicitly into characters, and supervise the
   model to reproduce its character sequence using simple original
   character-level tokenization.
3. Add deterministic *one-operation* tasks: append a letter,
   join parts, change an ending, then small morphological inflection.
4. Compare training on complete words versus letter-level and
   short-to-long teacher sequences with **matched update count**
   and separate train/test random strings and syllables.
5. Verify exact copy success, character error rate, first-token
   accuracy, output delimiter correctness and generalization to
   unseen strings, separately from language-model story retention.
6. Preserve the scratch-trained Forge model as the source. Any
   temporary character-level decoder or neural adapter must also be
   **original from random weights**, and should be evaluated before
   attaching it to the canonical language network.
7. Do not use the v2 exam again as an untouched test; create a
   fresh, independently generated transfer set for new training.

This is foundational English mechanics, not human-level literacy,
phonics, robust reasoning or conversational understanding. Android
development and all personal encrypted memories remain independent.

[Experiment specification](ENGLISH-PRODUCTION-V2.md).
