# Approved Forge Seed v0.7 core language continuation

Continue the original scratch-trained Forge Seed v0.7 language model from the
verified 28,000-step checkpoint created by Actions run 42085634121.

## Authorized run

- Perform 8,000 additional next-token optimizer updates on the original
  random-initialized Forge Seed model, resuming its optimizer and random state.
- Reuse the exact pinned TinyStories training and validation files and manifest
  stored with run 42085634121. Do not fetch a differently sized corpus or
  retrain a replacement tokenizer for this continuation.
- Verify the checkpoint step, seed, source revision/license, and SHA256 hashes
  before training; stop if any value differs.
- Preserve the parent checkpoint, candidate checkpoint, metric curve, samples,
  source manifest, and dataset split in a 90-day research artifact.
- Report validation loss before/after and basic held-out capability diagnostics.
  Do not call language-model loss a reasoning or intelligence score.

## Promotion / retention

This is a real continuation of Forge's core language model, not an auxiliary
role-study copy. Keep it as a research candidate until the held-out language
checks are reviewed. Do not replace the Android/app weights automatically.
Record the resulting run ID, checkpoint hash, metrics, and next action durably
in the experiment ledger after completion. No private data is involved.
