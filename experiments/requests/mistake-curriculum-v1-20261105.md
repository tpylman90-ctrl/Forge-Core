# Forge mistake-curriculum v1: authorized bounded study

Use the original scratch-trained Forge Seed v0.7 checkpoint at 76,000 optimizer
updates from Actions run 42123730003. The script verifies its checkpoint,
tokenizer, licensed TinyStories revision, and exact train/validation hashes.

Run nine matched candidates: three training arms by three independent sampler
seeds, 3,000 updates each, identical starting weights, optimizer state,
tokenizer, and source split.

1. Uniform shuffled replay control.
2. Model-surprisal replay, mixing verified high-loss training windows with
   ordinary random windows.
3. The same replay plus a 10% share of source-verified correction traces.
   Each trace retains Forge's wrong next-token guess, puts the correct answer
   before a short evidence statement, and derives its gold token only from the
   approved training text. This is a deterministic source oracle; no external
   pretrained teacher or private user data is used.

Use the exact same fixed 256-window validation protocol before and after each
candidate, report the existing 64-window diagnostic separately, and run the
common forced-choice capability check on all nine candidates. Keep every
candidate research-only and preserve lineage in the action artifact and draft
release. Do not alter Android weights.
