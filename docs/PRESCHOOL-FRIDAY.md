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

## November 3 direct-answer evidence

Controlled research on two copies of the original scratch-v0.7 checkpoint
trained a next-token **actual word** answer objective (rather than A/B).
The verified branch learned all 256 practiced cases and scored **9/20
original and 6/20 unfamiliar exact answers**, against **4/20 and
3/20** for shuffled answers. Its public-story held-out loss was
**2.661562**, compared with **2.637959** frozen. This is encouraging
for a small pilot but *below the preschool comprehension/instruction
gate*, with only twenty underlying cases per split. A separate source
check and held-out independent larger benchmark are needed before
any model promotion or capability claim. The human language criteria
remain open.


## Later Nov 3 measured gains (synthetic research, not a pass)

The 3,200-example compositional apprenticeship improved **direct answer
exact match** on a new internally authored 200-question challenge:
verified **49/200**, shuffled-answer control **13/200**, original
story-only model **0/200**. Giver and recipient questions were **0/40
each**; color 10/40, location 17/40, count 22/40. A separate one-step
instruction audit of 50 commands scored **11/50** verified versus
**4/50** shuffled; word repetition, list selection and recipient
commands were all 0/10. Public-language held-out BPE loss changed
2.637959→2.691088. The archived original remains unchanged.

This does **not** meet the provisional preschool instruction or
comprehension criteria. The next question is whether a targeted
copy/select and semantic-role curriculum can transfer to *new names*,
not whether training answers can be memorized.


## Teacher v1 / pointer results — no premature preschool certification

The new Teacher v1 pipeline generated verified, checksum-audited
lessons and development labels, and accepted only optional AI-produced
skill advice; no external AI teacher was executed in GitHub Actions.

The first teacher neural pilot was negative: parent **22/120**,
verified **19/120**, shuffled **14/120**. Frozen-backbone source-word
pointer training improved a 120-question development set to **42/120**
(vs untrained 7/120 and wrong-source control 4/120). However, a
new **360-question cross-template synthetic exam** produced only
**41/360** verified vs **12/360** control. The generalization
deficit persists; "preschool" is not an established capability.

All weights remain experimental and the original 14,000-step
scratch checkpoint remains preserved.
