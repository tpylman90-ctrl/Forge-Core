# Approved Forge Seed v0.7 core language continuation — round 2

Continue Forge Seed from the verified 36,000-step checkpoint produced by
Actions run 42115007553. Perform 8,000 additional next-token optimizer updates
on the exact same revision-pinned public train/validation split, retaining the
checkpoint's tokenizer, model, optimizer state, and RNG state.

Before training, verify the parent step, seed, source manifest/license, and
train/validation hashes. Stop on any mismatch. Evaluate held-out text loss
every 1,000 updates and at completion, preserve fixed generation samples, and
run the bounded capability diagnostic. Upload the parent and candidate
checkpoints plus data lineage as a 90-day research artifact.

This is approved core language-model learning. Keep the result as a research
candidate, record results durably, and do not automatically replace Android
weights. Interpret token-loss changes narrowly; require separate held-out
capability evidence before claiming improved comprehension or reasoning.
