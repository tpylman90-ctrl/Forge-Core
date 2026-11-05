# Approved Forge Seed v0.7 core language continuation — round 4

Continue from the verified 52,000-step Forge Seed checkpoint in Actions run
42118418883. Perform 8,000 additional next-token optimizer updates while
preserving its AdamW state, RNG state, tokenizer, model configuration, seed 43,
and exact revision-pinned public training/validation split.

Verify parent lineage, source license/revision and train/validation hashes
before training. Measure the same 64-window held-out progress curve every
1,000 steps and at completion, preserve fixed generation samples and run the
bounded capability diagnostic. Retain parent/candidate weights and all lineage
in a 90-day research artifact.

This is approved core language-model learning. Keep it research-only; do not
replace Android weights automatically. Continue only if the matched held-out
progress curve improves without a language regression. Do not treat token-loss
improvement as evidence of broad reasoning or comprehension.
