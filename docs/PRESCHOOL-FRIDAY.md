# Forge Seed v0.7 — Friday Preschool Readiness Gate

**Target checkpoint:** Friday, November 6, 2026 (US Eastern). Our goal is a
meaningful, measured improvement in early language. "Preschool" is a working
analogy, not a technical certification, and the deadline is a research
checkpoint rather than a guaranteed capability outcome.

## Verified starting point

The preserved original 1.875M-parameter Forge Seed was scratch trained on
800 TinyStories sample documents for 8,000 steps. It reduced held-out
next-byte loss, but six or fewer manually reviewed samples were not coherent.
Three separate linguistic-why experiments improved tiny agreement/negation
classifications but FAILED to learn subject/actor roles (50% on 320 held-out
questions). All candidate why-adapted weights remain experimental.

## What's changing in v0.7

1. **Immutable source reference.** Pin the real TinyStories upstream Git
   commit f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 and select bounded
   prefix samples from its original train and separate validation text files.
   This is better than a live, mutable dataset-viewer preview but samples
   a nonrepresentative prefix; do not claim a full-data run.
2. **Train our own tokenizer.** Train a 2048-token byte-level BPE vocabulary
   on the public training split only, rather than using another model's
   vocabulary. Retain exact tokenizer JSON, SHA256 and source manifest.
3. **Original neural weights.** Initialize a 4-layer, width-192 causal
   Transformer randomly for the new 2048-token vocabulary. Do not attempt
   a token-incompatible resume of the v0.6 byte-level checkpoint.
4. **Measured CPU learning.** Train 14,000 AdamW steps at batch 4,
   sequence length 128 learned BPE tokens, record loss every 2,000 steps,
   and keep a 256-window held-out audit and six fixed prompts.
5. **Rollback and privacy.** Experimental weights remain separate from
   archived v0.6 and from personal Android memory; no Android feature
   work or autonomous external actions.

## Friday capability rubric

Use **unseen** prompts and human-review generated completions, not just
falling next-token loss. We consider the preschool-like milestone met
only when at least the following are *independently demonstrated*:

| Skill | Provisional pass criterion |
| --- | --- |
| Sentence form | At least 8/10 prompted single-sentence continuations are readable, complete, grammatical, and relevant |
| Short story | At least 3/5 unseen prompts yield two linked sentences about the same characters or event |
| Basic facts / relations | At least 16/20 held-out, balanced who/what/color questions answered correctly (chance baselines reported) |
| Simple instructions | At least 8/10 unseen single-step instructions followed in the requested format |
| Consistency | No major collapse in held-out general-language loss versus the best suitable baseline |

Human and automated scoring must be described separately. Blind independent
rating is preferable for generated text. A yes/no or multiple-choice
measurement is NOT proof that Forge understands a concept; use novel
names, unfamiliar verbs, varied templates and counterbalanced answers.

**Current v0.7 run trains only next-token language.** The rubric includes
question answering and instruction following that require additional
vetted training, so do not expect all four skills from simple story
pretraining. If Friday arrives before the gate passes, report verified
improvements, remaining failures and a realistic next training plan.

## Fair science and compute

- Source: roneneldan/TinyStories; declared CDLA-Sharing-1.0; retain
  license notice. Original train text is ~1.92GB; stream just the first
  10 MiB; upstream separate validation text first 2 MiB. No private data.
- Raw pinned revision alone does not make the subset representative.
  Audit cross-document duplicates and eventual benchmark contamination.
- Avoid comparing BPE token cross-entropy directly with old byte cross-
  entropy; token granularities differ. The reported bits-per-byte estimate
  uses sampled windows and is not an exact cross-tokenizer comparison.
- GitHub Actions standard CPU runners are limited and billed/allocated
  per plan. One-shot bounded runs, not hidden indefinite training.
- Checkpoint promotion requires results across multiple seeds / sources,
  regression checks and user-visible sample review, not enthusiasm alone.

## Pending next steps

- [x] Revision-pinned bounded raw-corpus loader and deterministic unit tests
- [x] Train-only scratch ByteLevel-BPE with SHA256 and reversible-token test
- [x] Separate v0.7 scratch Transformer, resumable optimizer/RNG and
      fixed-prompt generation / held-out loss instrumentation
- [x] Verified 14,000-step v0.7 original scratch model and reviewed generated text; held-out token loss 7.6305 → 2.6303
- [x] Basic 40-choice mirrored question benchmark completed; 19/40 at chance level. Free-form instruction benchmark remains pending
- [x] Verified negative 1,600-example QA study and successful 16-example memorization sanity gate
- [ ] Run staged QA curriculum and assess novel question transfer; do not promote without evidence
- [x] Audited and created unpublished DRAFT model Release with tokenizer and source SHA256; not a publicly certified model
