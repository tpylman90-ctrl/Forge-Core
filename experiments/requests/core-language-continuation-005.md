# Approved Forge Seed v0.7 core language continuation — round 5

Continue from the verified 60,000-step Forge Seed checkpoint in Actions run
42120062062. Perform 8,000 additional next-token updates, preserving AdamW
and RNG state, tokenizer, architecture, seed 43, and the exact pinned public
train/validation split. Verify all hashes, source revision/license and parent
step before training; stop on any mismatch.

Measure the 64-window progress curve every 1,000 steps and the same 256-window
final diagnostic, plus the bounded capability check. Preserve the parent and
candidate checkpoints, source split, metrics and lineage in both the bounded
Actions artifact and an unpublished research draft archive.

This is approved core language-model learning. Keep all research-only and do
not change Android weights automatically. Continue only if the matched
held-out signals improve without regression; do not equate language loss with
comprehension or general intelligence.
