# Forge English v5 — Independently curated word-source transfer audit

**Registered during the v5 CPU training run, before inspecting its final
results. No model training or checkpoint promotion.**

The v5 main exam tests 40 fresh strings of uniformly random letters,
each paired with a one-character-changed twin. That removes the old
alternating vowel/consonant artifact. But random strings are not an
English literacy test, either.

This follow-up evaluates **the already trained and
development-selected v5 policy weights** on 36 manually curated
human-readable English source words, each paired with a deterministic
one-character-changed counterpart. Words range from 4–8 letters and
include `apple`, `cloud`, `island`, `garden`,
`thunder`, and `dolphin`. Every source index is explicitly
queried; no word is submitted to a model for training.

The word list was authored by an AI assistant, **not externally
certified or human-authored**, and may contain words present in
original public TinyStories pretraining. That matters: the
one-character-edited *nonword twins* are a stronger source-dependence
test than familiar originals alone.

## Controls and boundaries

- Load the exact original v0.7 Forge Seed weights and tokenizer,
  audited with SHA256 and the original public training manifest.
- Load exactly the eight **already selected** experimental v5
  checkpoints from Actions run 41886375209 (four conditions,
  training seeds 11631 and 11632), with no optimizer step.
- Reject any mismatch in source checkpoint SHA, tokenizer SHA,
  experiment manifest, research-only flag, arm or seed.
- Programmatically verify that no source word or twin appeared
  in either seed's v5 train set or in the v5 development or
  final synthetic exam sets.
- Use the identical original text prompt format and independently
  reconstruct the correct character from `source[index-1]`.
- Score exact individual-letter answers, counterfactual both-correct
  pairs, whether twin predictions change, and complete strings
  **externally assembled** from separately requested letters.
- Preserve the source-masked arm as a source-blind control.
- The exact symbolic source checker should obtain 100% here,
  but this is merely a reference, not earned neural learning.
- Record the frozen original-model baseline and all four trained
  arm outcomes separately by seed.

A successful result would corroborate **source-position retrieval**
on less artificial text. It still would not demonstrate the
meaning of a word, autonomous copying, spelling without indexed
questions, sentence comprehension or preschool proficiency.

All research model states remain unpromoted; no Android signing,
mobile code or private user memories are read or modified.

Source:
`src/forge_core/neural/english_v5_word_challenge.py`.
Tests:
`tests/test_english_v5_word_challenge.py`.
Workflow:
`.github/workflows/english-v5-word-challenge.yml`.

The audit must not silently retrain on these words, and its result
should not be reused as an untouched exam in a later iteration.
