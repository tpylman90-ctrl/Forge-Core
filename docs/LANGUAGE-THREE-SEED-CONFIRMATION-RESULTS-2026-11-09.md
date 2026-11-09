# Language three-seed confirmation — 2026-11-09

## Purpose and lineage

This was a controlled confirmation of the latest Forge Core language continuation, not a capability certification. Each arm restored the same verified 106,000-update Forge Seed parent and used the same model, tokenizer, TinyStories train/validation/test slices, optimizer state, sequence window, learning rate, and evaluation procedure. Only the sample-order seed differed. Each arm selected its checkpoint by validation loss and was then compared on the fixed 2,048-token-window independent test set.

- Parent checkpoint SHA-256: `b038869062c3b051768b10b1f146aeb8fdde7419cfdd2e8ebdf54acfbc2ea42b`
- Parent run: [42445321887](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42445321887), step 106,000
- Training-data SHA-256: `81e8ef34fc61996f8cf0e783c95daa5521074b3ec8b9a7db4912445a6bb717cd`
- Validation-data SHA-256: `ed86a376fb836f86829b930857e7ce25fd70c8824c12d48c558af854154d88ef`
- Independent test SHA-256: `9d3ad642916b7b4494efae808ce6683b21e8b8eed3778ade934cf2924b86ea0f`
- Tokenizer SHA-256: `4ecd7803d4c369e80bda2e74c30cfb769b7e3a942f931df83383eec9f794f04c`

## Results

| Requested sampler seed | Updates | Selected total step | Validation loss, parent → selected | Independent test loss, parent → selected | Selected checkpoint SHA-256 |
|---:|---:|---:|---:|---:|---|
| 43 | 1,000 | 107,000 | 2.00455068 → 2.00333783 | 1.96600281 → 1.96508341 | `ed2f1395b52c3ae2c674e00f3b86f82ec26685e2d1c631f6eb975ddb3f05fe0f` |
| 73 | 500 (early stop) | 106,000 parent retained | 2.00455089 → 2.00455089 | 1.96600302 → 1.96600302 | `b038869062c3b051768b10b1f146aeb8fdde7419cfdd2e8ebdf54acfbc2ea42b` |
| 101 | 1,000 | 106,750 | 2.00455068 → 2.00375978 | 1.96600281 → 1.96534054 | `cc1c48cee0f998dbd006f4caef82e4f782c340c9df95c470a480a15a182a7ae4` |

For seed 73, the selected checkpoint remains the parent because the candidate did not beat validation loss; its last in-run checkpoint had worse test loss (1.96683982). The other two seeds both improved the fixed independent text-prediction loss by small amounts. This supports modest repeatable text-loss progress for two of three short continuations, not a general reasoning or comprehension claim.

## Capability probes

The fixed language forced-choice probe returned 20/40 for the parent and each selected checkpoint. Across 20 mirrored pairs, zero pairs had both answer-order orientations correct (20/20 had exactly one correct orientation). The fresh A/B probes also returned 20 correct for the parent and selected candidates. These results do not certify capability; there was no human grading of generated stories or answers.

## Sampler metadata correction

The requested seed 43 run did use a distinct explicit sampler seed, `manual_seed(44)`, and saved its RNG state. However, the run's reported `sampler_rng_policy` label incorrectly said `resumed_parent_rng` because the reporting condition still treated seed 43 as the legacy resume case. The training code has since been corrected so only seed 0 resumes the parent RNG, and the metadata label now matches the actual behavior. This correction does not alter the archived checkpoint or its measured results.

## Preservation and status

All three job bundles, including parent and selected/last checkpoints, optimizer state, tokenizer, pinned data, probe outputs, and hashes, were archived in the successful [workflow run 42539881351](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42539881351) and draft release `forge-language-confirmation-42539881351`. These candidates remain research artifacts and were not promoted to Android or substituted for Forge's from-scratch weights.
