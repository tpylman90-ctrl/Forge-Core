# Forge 95k vs 152k Independent Language Audit — 2026-11-06

## Result

The 152k uniform-language research candidate did **not** beat its verified 95k parent on the independent TinyStories sample.

| Checkpoint | Step | Independent test token loss |
|---|---:|---:|
| Verified parent | 95,000 | 3.37113519 |
| Isolated language candidate | 152,000 | 3.40505678 |

Candidate minus parent is **+0.03392159** (about 1.0% higher; lower is better). The earlier reported 2.6% validation improvement does not hold on this untouched slice.

Both checkpoints scored **20/40** on the new mirrored forced-choice set, with **0/20 pairs** fully correct in both orientations. This is at chance on a narrow diagnostic. The ungraded generation samples remained story-like but did not follow several prompts. Human grading was not performed.

## Independent data and reproducibility

- Dataset: TinyStories, revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, license CDLA-Sharing-1.0.
- The original training prefix is the first 52,428,800 source bytes. The audit used the following 4 MiB range, bytes 52,428,800–56,623,103.
- After removing incomplete boundary records and exact duplicates against train and validation, the fixed test set contains 4,162 stories, 3,885,993 UTF-8 bytes, and 1,078,052 tokens.
- The audit used 2,048 evenly spaced windows of 256 tokens with the same verified tokenizer for both checkpoints.
- Raw range SHA-256: `fd92237646d4c1de4f68bb3745a05afb5d3a3d5eba3d8feb89da516e0af54f07`.
- Clean test SHA-256: `2f816a0fde9e8b6960c6e51673ec9286b6438d9866f4678e9535cfbcbdc4609e`.
- Parent checkpoint SHA-256: `a7bcdb996e440f44d7496b887b54d19324927b4296fe9ed2f0f0293c88c63a0a`.
- Candidate checkpoint SHA-256: `1400ab3acd5ff0fe206829d5c92418a43d355ddb9ad7e9c4187fc5c46d2a1965`.

## Evaluation correction

The first multilane language run copied the validation file to the test file, so its reported test score was not independent. The loader now fetches a pinned range beyond the training prefix, checks HTTP range behavior and exact duplicates, writes a separate test manifest/hash, and rejects a test/validation hash match. The audit workflow applied this corrected source logic to the already archived 95k and 152k checkpoints; it performed no optimizer updates.

## Decision

Keep 152k as a research artifact and do not use it as Forge’s next continuation parent. Continue from the verified 95k checkpoint with the corrected independent test loader. Neither checkpoint is certified for comprehension or general reasoning, and neither is integrated into Android.

Audit run: https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42231310241
Draft audit archive: https://github.com/tpylman90-ctrl/Forge-Core/releases/tag/forge-language-independent-audit-42231310241
