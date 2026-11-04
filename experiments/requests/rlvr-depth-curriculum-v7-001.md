# Forge RLVR v7: one-time increasing-depth formal proof pilot

This request launches one strictly bounded neural research experiment
after CI validates its exact depth checks.

- 900 updates per arm and independent seed (5321, 5322).
- Same scratch-started proposal network for all three conditions:
  flat 2–3-step verified imitation, staged 1→2→3-step verified
  imitation, and staged imitation with small on-policy RLVR.
- Retain 50/50 solvable/impossible and arithmetic/logic balance in
  all curriculum stages. Preserve exact formal execution.
- Select checkpoint with development-only results. Evaluate new
  procedural seeds at proof depths 1, 2 and 3 separately, including
  shifted arithmetic ranges and irrelevant symbolic premises.
- One hosted GitHub CPU runner, serialized with other RLVR studies,
  25-minute job timeout, no recurring training or unbounded stress test.
- Store full metrics, experimental best/last policy weights, optimizer
  state and RNG provenance.
- Never modify Android files, the original Forge Seed language
  model, pretrained teacher weights or user's encrypted memory.
- Do not auto-promote a trained policy; any capability claim requires
  independent novel tasks and genuinely verified full proofs.

The flat model is an equal-update curriculum control; the small
RLVR term must not be misrepresented as all of the supervised learning.
