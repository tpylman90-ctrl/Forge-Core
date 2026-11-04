# Forge English v3 — Letter copying as a prerequisite to word meaning

## Motivating evidence

In the measured [v2 generative English study](ENGLISH-PRODUCTION-V2-RESULTS.md),
eight original Forge Seed adaptations across two seeds each scored
**0/112** exact new words/sentences on an unfamiliar synthetic exam.
The verified-teaching branches reduced their teacher-forced lesson loss,
but produced familiar training responses instead of copying unseen
source content. Examples include answering a plural `pillow`
question with `windows`, or an unrelated old sentence rather
than a new Soren/box sentence.

That is a concrete *content-binding and spelling-copy* problem.
More full-sentence supervision is unlikely to fix it without an
ability to retain and manipulate new written character sequences.

## Four stages, same original language model

V3 reuses the **same approved scratch-trained Forge Seed v0.7
Transformer and BPE tokenizer**, without introducing another
pretrained language model or independently deploying an additional
assistant.

The teaching exercises are small original letter-sequence operations:

| Stage | Question source | Exact expected output |
| --- | --- | --- |
| Copy one letter | `letter=k;context=cat` | `k` |
| Copy short sequence | `letters=c a t` | `c a t` |
| Join letters | `letters=c a t` | `cat` |
| Append a letter | `letters=c a t;append=s` | `cats` |

The output is **generated** and terminated by a newline. No A/B
options or reference correct next-token IDs are included as input.
Independent explicit regex-based oracles verify the input and
answer. Full answer tokens only are supervised in training; the
question context is not a target.

Each source uses synthetic lowercase, English-like alternating
consonant/vowel sequences. To avoid memorized whole answers,
all **multi-character target strings** are assigned to a train,
development, or exam split using a fixed SHA256 bucket. Letters
themselves necessarily repeat across all splits; copied single
letters are tested in novel written contexts. The final exam is
generated with a different seed and uses different wording, though
task structure remains the same.

## Three different learning questions

1. Can Forge **copy the alphabet** as generated text rather than
   merely recognize letter names?
2. Can it copy and join **whole previously unseen letter sequences**?
3. Does a staged curriculum learn that skill better than exactly
   equal-exposure mixed lessons?

Four arms start from identical original source model weights:

- `verified_staged`: copy-letter → spaced-copy → join → append,
  while retaining one earlier-category practice item per batch;
- `verified_mixed`: the exact same multiset of training batches,
  randomly shuffled in time;
- `shuffled_staged`: identical staged input prompts, but wrong
  answers permuted within task categories;
- `story_only`: the same optimizer update count, training only on
  the original authenticated public TinyStories corpus.

Teaching arms use **20% original-story rehearsal** and 80% letter
copying loss; the story-only arm receives only source-text training.
No original public training document or validation sample changes.

## Measures

All results are from **greedy model-generated text** on never-trained
strings:

- Exact output accuracy, separately per skill;
- First generated character correctness;
- Normalized Levenshtein character error rate (edits / target chars);
- Newline-answer termination rate;
- Preview examples of generated text;
- Held-out original TinyStories next-token loss, using identical
  48-window, 64-token settings across all arms.

This is important because a model may make a real improvement in
individual letters without completing long words exactly.
Only a strong exact-copy rate on unfamiliar sequences is evidence
of transferrable written content copying.

Use development data only to choose checkpoints, primarily by
exact output count, then first-character correctness and aggregate
edit distance. The final novel synthetic exam is used after
checkpoint selection, never for choosing hyperparameters. Once
inspected, it is no longer a sealed independent exam.

## CPU budget and isolation

Two independent training/schedule seeds, **9231 and 9232**,
default **120 optimizer updates per four stages** (480 per arm
per seed), four sequential comparison conditions on one hosted
CPU GitHub Actions runner. Bound the full job to 45 minutes
including installation, CI verification, public artifact download,
inference and upload. No recurring computation.

The exact original checkpoint from run **41572257363** remains
frozen and protected. Trained branches are **temporary research
artifacts** with source lineage, optimizer/RNG state, hashes and
examination reports. Nothing is promoted to canonical Forge Seed,
Android, personal memory or autonomous action systems.

A letter-copy improvement would be foundational orthographic
skill, **not auditory phonics, reading comprehension, spoken
language or independent reasoning**. Later stages must combine
reliable copying with grammar and natural text while checking
language retention.

[The underlying v2 negative evidence](ENGLISH-PRODUCTION-V2-RESULTS.md).
