# Forge English v6 — One-time original neural copy-component experiment

The user's standing approval is to implement research changes that
may materially improve Forge's English language training, while
keeping the original scratch-trained identity and Android app safe.

## Verified motivation

In the completed v5 training and independent word challenge, correct
labelled instruction produced **173/1016** individual characters on
random sources and **166/856** on English word/nonword sources. But it
completed only **3/80** hard changed-letter pairs in the synthetic
test and **no** complete externally assembled words. More ordinary
BPE language-supervision repetitions are not the right next test.

## Scope

- Create an **original randomly initialized auxiliary character-aligned
  neural decoder**; do NOT replace or fine-tune the canonical
  TinyStories-trained Forge Seed v0.7 language model.
- Train the module to generate *whole* source strings and a learned
  EOS from its own predicted prior characters, with no source-position
  question loop or deterministic post-hoc character assembly.
- Compare five arms: learned source attention with exact alignment
  supervision; identical attention model with token loss alone;
  pooled-source decoder without character-slot attention; attention
  with source letters masked in both training and inference; and
  attention trained against deliberately wrong same-length labels.
- Each arm sees **800 optimizer updates** on **512 source families**
  per training seed, 8 source/twin words per step, using training
  seeds 13631 and 13632. One sequential CPU GitHub Actions job
  with **45-minute maximum duration**, no automatic retry or
  recurring workload.
- All synthetic letters are sampled independently from a–z. Test
  on new, disjoint families 3–8 letters, extra-long **9–12 letter**
  sources never seen at training lengths, and 36 separately listed
  familiar English words + altered nonwords. Test cases are never
  selected for checkpoint promotion.
- Select weights on development-only exact full-word and changed
  twin metrics. Report true unassisted whole-string generated
  output, Levenshtein errors, both twins correct and changed
  letter position success.
- Validate the original Forge Seed v0.7 checkpoint and tokenizer
  source SHA256 from Actions run **41572257363**, but **do not
  copy its parameters into this auxiliary module**. Its language
  retention is guaranteed by keeping it frozen/untouched, not
  demonstrated by the aux module's isolated test.
- Archive only experimental research weights, complete metrics and
  their source/seed provenance. **No checkpoint promotion.**
- Do not touch `android/`, APK signing/keystores, private local
  memories, RLVR research or external action permissions.

Success is not assumed. If the pointer method learns complete copying
on in-distribution strings but fails on longer new strings, do not
claim full language literacy. Any neural integration into Forge's
main BPE network requires a new, separate exact-answer and
language-retention experiment.
