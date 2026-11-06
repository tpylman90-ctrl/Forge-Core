# Approved matched language candidate audit

Evaluate the existing checkpoints only; do not run optimizer updates and do not
promote or modify either model.

- Parent: run 42143390701, Forge Seed step 95,000, checkpoint SHA-256
  a7bcdb996e440f44d7496b887b54d19324927b4296fe9ed2f0f0293c88c63a0a.
- Candidate: language lane in run 42149964829, step 152,000, checkpoint SHA-256
  1400ab3acd5ff0fe206829d5c92418a43d355ddb9ad7e9c4187fc5c46d2a1965.
- Use the identical verified tokenizer and the same fixed windows for both models.
- Reserve the next 4 MiB after the original 50 MiB TinyStories training prefix at
  the same pinned revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 as a genuinely
  untouched language test sample. Verify the server honored the byte range, remove
  incomplete boundary records, screen exact overlap with train and validation,
  preserve source/data hashes and attribution.
- Run 20 newly written mirrored fact questions (40 orientations) with identical
  prompts for both models; include per-skill and paired-orientation scores.
- Save ungraded generation samples and make no human-quality or capability claim.
- Preserve the report, independent test data, lineage, and hashes as a draft research
  artifact. Keep all weights research-only and separate from Android.
