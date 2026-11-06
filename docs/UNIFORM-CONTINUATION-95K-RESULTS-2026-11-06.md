# Forge Seed v0.7 uniform continuation — 87k to 95k

Date: 2026-11-06

## Run and lineage

- Correct run: [GitHub Actions run 42143390701](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42143390701)
- Parent run: 42138215923, verified 87,000-step uniform checkpoint
- Candidate: 95,000 steps after 8,000 ordinary uniform next-token updates
- Training: CPU, seed 43, batch 4, context 128, learning rate 0.0003
- Model: 2,219,136 parameters, scratch-initialized Forge Seed v0.7, resumed from its own checkpoint and optimizer state
- Data: TinyStories revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, license `cdla-sharing-1.0`
- Train SHA-256: `81e8ef34fc61996f8cf0e783c95daa5521074b3ec8b9a7db4912445a6bb717cd`
- Validation SHA-256: `ed86a376fb836f86829b930857e7ce25fd70c8824c12d48c558af854154d88ef`
- Tokenizer SHA-256: `4ecd7803d4c369e80bda2e74c30cfb769b7e3a942f931df83383eec9f794f04c`
- Parent checkpoint SHA-256: `3a8504ce39b0da5bd444bd472b12f327b55d30d2c55ed040b86b2af1de247814`
- Candidate checkpoint SHA-256: `a7bcdb996e440f44d7496b887b54d19324927b4296fe9ed2f0f0293c88c63a0a`
- Durable draft release: [95,000-step research checkpoint](https://github.com/tpylman90-ctrl/Forge-Core/releases/tag/untagged-890157899bb03288666e)
- Actions artifact digest: `sha256:bb1a29e3d65763292a0b7f3424aa10e3432869a768e163b55ec249485290f6ba`

## Matched held-out result

The same tokenizer, validation split, and 2,048 evenly spaced 128-token windows were scored for both checkpoints.

| Checkpoint | Token loss |
|---|---:|
| 87k parent | 2.13098588 |
| 95k candidate | 2.11754341 |
| Change | -0.01344247 (-0.63%) |

This is a modest improvement in next-token prediction on this fixed sample. It does not establish comprehension, reasoning, or instruction-following. The separate training report's 64-window curve and 256-window end diagnostic are different samples; do not mix those values with the matched 2,048-window result.

## Capability diagnostic

- Forced choice: 20/40 (50%)
- Fully correct mirrored pairs: 0/20
- Human sentence/story grades: not performed
- Certification: not certified

The fixed forced-choice check remained at chance. There is no evidence here of a capability gain. The candidate remains research-only and is not integrated into the Android app.

## Workflow lineage incident

Run 42141780279 completed training but resumed from the 68,000-step checkpoint in run 42121807463, yielding a 76k candidate. Its push-trigger expression used the old fallback run ID even after the request named the 87k parent. That artifact is not part of the 87k-to-95k continuation and must not be treated as its parent or candidate.

The workflow's push path now explicitly selects run 42138215923, and the corrected run's artifact lineage verifies parent run 42138215923, parent step 87,000, and candidate step 95,000.

## Disposition

The corrected training, matched evaluation, capability check, artifact upload, and durable draft archive all completed successfully. This is a resumable research checkpoint only; it was not promoted to Android or certified as a capability upgrade.
