# Linguistic Why Lab — first controlled A/B experiment

Authorized one-time GitHub Actions research run as part of Forge Core's
language-first development. This document is a trigger marker, not a schedule.

- Original checkpoint: public-data Forge Seed Run 005, ID 41558008416,
  8,000 steps, 1,875,072 parameters, randomly initialized originally.
- Copy weights into two identical independent adaptation arms, starting
  with newly initialized AdamW optimizers.
- 120 updates, batch size 2, same prompt order and training opportunities.
- Arm 1: original verified grammatical/semantic labels.
- Arm 2: within-skill class-balanced shuffled labels (negative control).
- Compare a third *frozen* reference without training.
- Measure held-out A/B choices on unseen nouns **and verbs**, plus
  held-out public TinyStories next-byte loss regression.
- Outputs are experimental only, not promoted or integrated into Android.
- Personal memory, observation logs, account identifiers and private
  conversation data are excluded.
