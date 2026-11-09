# Forge language large independent audit — 2026-11-09

## Status

Evaluation-only comparison of the verified 107,000-step language parent and its selected 107,250-step continuation. This audit performed no optimizer updates and did not promote a checkpoint.

## Lineage

- Parent: run 42552572467, step 107,000, SHA-256 `ed2f1395b52c3ae2c674e00f3b86f82ec26685e2d1c631f6eb975ddb3f05fe0f`
- Candidate: step 107,250, SHA-256 `848553a81549c1269c360c5d07c57adde92a7816a0bc19b61ecc8028c93fee99`
- Audit run: [42672695518](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42672695518)
- Archived audit bundle: [draft release](https://github.com/tpylman90-ctrl/Forge-Core/releases/tag/untagged-2f92c98f0bb80f2ea6cc)

## Data and method

- TinyStories revision: `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`
- Dataset license metadata: `cdla-sharing-1.0`
- New raw sample: 16 MiB beginning at byte offset 24 MiB beyond the pinned training prefix; it is disjoint from the earlier 20–24 MiB test range.
- Exact duplicate stories against training, validation, and the earlier test were removed.
- Cleaned test: 4,467,651 tokens across 18,757 documents; SHA-256 `f9ec3e522cfdabb890271c4ba4ca792f79a788c110872ea0eedc9e04a636c199`
- Raw byte range SHA-256: `33dff80ae51ae392441ebb48609deb8abd0e749024faf89a15cb4de1db210218`
- Same deterministic evaluation for both checkpoints: 4,096 uniformly spaced windows, 128 tokens of context.

## Result

| Checkpoint | Held-out next-token loss |
|---|---:|
| Parent, 107,000 steps | 1.99253691 |
| Candidate, 107,250 steps | 1.99189755 |
| Candidate minus parent | -0.00063936 |

This is a modest next-token prediction improvement on one larger independent sample. It does not demonstrate comprehension, reasoning, or general capability. The candidate remains research-only.