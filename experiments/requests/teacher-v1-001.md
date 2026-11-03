# Forge Teacher v1 — bounded initial lesson and student research

Owner approves using an optional existing AI as a TEACHER, not as a
replacement for Forge's original randomly initialized neural network.

The architecture is free/open-source, local-first and privacy-preserving:
- The existing local llama.cpp/GGUF adapter can suggest one curriculum
  focus from aggregated *development* scores, never user observations.
  It is optional and is NOT invoked in this hosted GitHub CPU run.
- A deterministic Forge Core rule-based validator generates all actual
  lesson prompts and gold labels. The teacher cannot alter them.
- Training pool has 240 selected original nonpersonal questions across
  copying, list selection, giver, recipient, color and quantity.
  Development has 120 separately generated questions with disjoint
  training people/words, kept out of gradient updates.
- Download only audited original Forge Seed v0.7 run 41572257363 and
  the verified compositional research branch from run 41649770774.
  Verify checkpoint SHA256, architecture and source manifest.
- Independently adapt two copies from the identical parent neural
  research weights: verified gold answers vs within-skill shuffled
  answer controls, matched 280-step budgets, batch 6, equal 25%
  public TinyStories next-token rehearsal.
- Compare aggregate development exact-word results and held-out public
  story loss. No external natural-language capability certificate is
  implied by this development benchmark.
- Generate a next-lesson bundle based on *aggregate* development counts,
  but do NOT train that second bundle without the next controlled study.
- All data and weights remain experimental research artifacts; do not
  publish a model, modify user memory, enable autonomous actions,
  or integrate experimental checkpoints into Android.
