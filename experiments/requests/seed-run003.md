# Forge Seed Run 003 — One-time authorized CPU continuation

Owner-approved language-first development experiment. This marker intentionally
triggers **one** bounded GitHub Actions run; the workflow has no schedule.

- Starting checkpoint: Forge Seed Run 002, workflow `41554336734`,
  20 optimizer steps, original randomly initialized 1,875,072-parameter model.
- Corpus: exact-content locked public TinyStories viewer preview, 200 train
  documents / 50 validation documents; refuse hash mismatches.
- Additional training: 1,000 AdamW steps, seed 42, batch 2, sequence length 64.
- Evaluation: held-out loss every 100 steps; wider validation audit;
  fixed-seed generation sample. Report failures and regressions honestly.
- Retention: public-source training/validation split snapshots, checkpoint,
  optimizer, RNG and source manifest in 30-day GitHub artifact.
- Purpose: demonstrate whether the **original Forge network** makes sustained
  progress on actual text. This is not proof of fluent language or independent thought.
