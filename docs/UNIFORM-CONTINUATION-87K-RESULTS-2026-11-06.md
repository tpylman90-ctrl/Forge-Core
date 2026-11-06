# Forge Seed v0.7 uniform continuation results — 2026-11-06

## Run

- GitHub Actions run: 42138215923
- Parent: run 42133325924, uniform arm, sampler seed 73, step 79,000
- Candidate: step 87,000 after 8,000 ordinary shuffled next-token updates
- Training seed: 43; batch size 4; context length 128; CPU
- Data: pinned TinyStories revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, license `cdla-sharing-1.0`
- Source files, tokenizer, checkpoint lineage, and evaluation are in the run artifact and draft release.
- Candidate checkpoint SHA-256: `3a8504ce39b0da5bd444bd472b12f327b55d30d2c55ed040b86b2af1de247814`

## Results

| Measure | 79k parent | 87k candidate | Change |
|---|---:|---:|---:|
| Held-out token loss, same 256 windows | 2.154432 | 2.136017 | -0.018415 (-0.85%) |

The candidate improved the matched text-loss measure. This is a small signal on one fixed 256-window sample, not proof of broad language or reasoning gains. The training script's periodic curve uses a different 64-window diagnostic; its displayed starting value (2.128734) must not be compared with the 256-window parent measurement.

## Capability check

- Forced-choice: 22/40 (55%)
- Mirrored pairs: 2/20 with both orientations correct; 18/20 with one orientation correct
- Certification: **not certified**. The configured mirrored-choice threshold is 32/40. Human sentence/story grading was not performed.
- This check is a narrow diagnostic. It does not establish general comprehension, independent reasoning, or instruction following.

## Disposition

The CPU training job, fixed-window comparison, capability check, artifact upload, and durable draft archive all completed successfully. This is a research candidate and resumable training checkpoint only. It was not promoted to Forge's Android app or treated as an approved capability upgrade.
