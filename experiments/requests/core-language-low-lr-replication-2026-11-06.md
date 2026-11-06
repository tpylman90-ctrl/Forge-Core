# Approved low-rate continuation seed replication

Run two additional language-training arms (sampling-order seeds 73 and 101)
to replicate the completed seed-43 conservative continuation.

- Both arms must start from the exact verified 95,000-step Forge Seed parent,
  SHA-256 `a7bcdb996e440f44d7496b887b54d19324927b4296fe9ed2f0f0293c88c63a0a`.
- Resume Forge's original weights and AdamW moments. Change only the learning
  rate from 0.0003 to 0.0001 and the sample-order RNG; do not reinitialize or
  import teacher weights.
- Use the pinned TinyStories revision, same tokenizer, batch 4, context 128,
  and at most 4,000 updates. Evaluate the same 2,048-window validation split
  every 1,000 updates; early stop after two checks without a 0.0001 improvement.
- Select using validation only. Report the untouched test-range loss for parent,
  selected candidate, and last attempt, plus both fixed capability diagnostics.
- Preserve parent/candidate checkpoints, optimizer/RNG, hashes, exact source
  data, curves, metrics, and lineage in per-seed Actions artifacts and draft
  research releases.
- Keep all outputs research-only. Do not promote or integrate any candidate.

The seed-43 run improved fixed held-out language prediction, but capability
probes stayed at chance. These two runs test whether the loss improvement
replicates under different sample orders; they cannot certify comprehension
or reasoning.
