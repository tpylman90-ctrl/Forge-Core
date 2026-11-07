# Approved 101k low-rate follow-up

Continue from the selected validation-best 101,000-step Forge checkpoint in
run 42290966213, artifact `forge-seed-followup-99k-42290966213`.

- Verify parent checkpoint SHA-256
  `94d87368faf8bad7fb34a43ca75448ddcd96bf2079dd731d4b5a0b808baf2eda`,
  global step 101,000, optimizer learning rate 0.00005, and sampler seed 73.
- Resume Forge's original student weights, optimizer moments and saved sample
  RNG. Preserve tokenizer and exact train/validation text.
- Run uniform TinyStories next-token training only at learning rate 0.000025,
  for at most 2,000 updates. Evaluate the same 2,048-window validation split
  every 500 updates; stop after two consecutive checks without at least 0.0001
  validation-loss improvement.
- Use a second fresh pinned TinyStories test range (8 MiB offset after the
  training prefix, after the prior 4 MiB test range). Filter exact train and
  validation duplicates and record the new byte range and hashes. Use it only
  for final parent/candidate comparison.
- Archive parent and both selected/last candidate checkpoints, optimizer/RNG,
  data, hashes, curves, capability probes and lineage in Actions artifacts and
  an unpublished draft release.
- Keep all results research-only. No Android promotion or integration.

The previous 99k-to-101k follow-up improved both validation and a new
independent test slice. This run checks whether the signal continues at half
the learning rate; it is not a capability or reasoning claim.

Run a matched 2,000-update replication from the exact 101k parent using sampler seed 101 (the previous run resumed seed 73). Preserve validation/test windows and all hashes; candidate remains research-only.
