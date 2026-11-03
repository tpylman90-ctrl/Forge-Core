# Forge Seed Run 005 — Bounded continuation on frozen expanded corpus

Approved project training iteration. One push-triggered GitHub workflow run
only; no ongoing automatic schedule.

- Resume Run 004 successful checkpoint: `41557239198`.
- Model: unchanged 1,875,072-parameter randomly initialized Forge Seed.
- Dataset: same **exact-content locked** 800/200 public TinyStories split,
  file hashes and manifest verified against Run 004.
- 4,000 additional AdamW CPU steps, global step target 8,000.
- Batch 4, sequence 128 bytes, seed 42, validation every 400 steps.
- Wider 256-window held-out audit; generated text is evaluated honestly.
- Save checkpoint with optimizer/RNG, source notice and public corpus.
- Do not update private memory, Android observation features or pretrained
  model weights.
- If held-out validation worsens, retain Run 004 as the better baseline.
