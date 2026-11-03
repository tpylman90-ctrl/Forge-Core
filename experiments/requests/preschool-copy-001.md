# Forge Seed v0.7 — Copy + semantic-role apprenticeship, one-time run

In the verified 3,200-example compositional research run, direct answers
improved to **49/200** held-out versus **13/200** for shuffled answers.
Per skill: count 22/40, location 17/40, color 10/40, but both unfamiliar
giver and recipient identification remained 0/40. A separate unseen
50-instruction audit: verified 11/50 vs shuffled 4/50, with 0/10 word
repetition, 0/10 list selection and 0/10 recipient instructions.

We hypothesize missing **word/name copying and selection** is a
prerequisite to accurately handling roles for unfamiliar people.

One-time, bounded research experiment under the Forge Core development
authorization:
- Preserve original random-initialized 14k-step source checkpoint
  (run 41572257363) and its public TinyStories tokenizer/training corpus.
- Read *only* the verified research candidate from compositional run
  41649770774; validate its source SHA256 and matching model architecture.
  Two identical independent models branch from these weights.
- Train explicit commands for six skills: copy word/name, select list
  entry, identify recipient, identify giver, answer a color, answer count.
- 120 → 600 → 2,400 original rule-derived training examples,
  with 250 → 600 → 1400 AdamW updates PER ARM.
- Gold-answer words vs skill-frequency-preserving shuffled answers;
  identical prompts, learning rates, public TinyStories rehearsal and
  sampling schedules, with no private data or pretrained teacher labels.
- New 120-question **held-out** exam uses disjoint names/words and
  unseen sentence/command templates. Previously scored 50-instruction
  and 200-QA suites are used only as regression diagnostics.
- Save held-out word-answer exact matches and per-skill errors,
  prior QA retention, story-language loss and fixed-seed continuations.
- Neither candidate is promoted, integrated into Android or described as
  preschool-capable without further independent replication.
