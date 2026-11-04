# One-time English Class v2 — actual written-answer learning

This request authorizes one controlled original Forge Seed experiment.
It is independent of the separate Android/mobile conversation.

- Extend the six-level first English Class with actual **generative
  written answers** and a seven-skill curriculum: joining word parts,
  plural nouns, present verb agreement, regular past tense, actor versus
  receiver, full positive sentences and full negative sentences.
- Start each learning condition from the identical, approved original
  scratch-trained Forge Seed checkpoint and tokenizer from public-only
  v0.7 training run 41572257363. No pretrained weights.
- Compare verified mixed lessons, verified staged lessons, wrong-answer
  shuffled lessons, and a public-story-only equal-update control.
- Two independently seeded training schedules **8123 and 8124**,
  80 optimizer updates per skill level, seven levels:
  560 updates per arm per seed; eight sequential model adaptations
  on a single public GitHub Actions CPU runner, total 45-minute
  job timeout, with no recurring workload.
- Preserve the original source and reference held-out story text.
  Allocate 25% of each grammar-training step to public-story rehearsal.
- All results must be exact **greedy generated words/sentences** without
  A/B answer options, not teacher-forced training accuracy alone.
- Select model weights on development-only tasks; use a newly authored
  **112-case synthetic exam** with lexically independent names,
  nouns, verbs and written parts. Record individual skill counts,
  generated examples, null/control scores, and matched 48-window
  TinyStories retention loss.
- No experimental weights are promoted to the canonical model.
  No changes to Android, its APK signing, device observations or
  the user's encrypted personal memory.

High training loss improvement or a few lucky examples do not count
as independent English comprehension. Negative results are valuable.
