# Forge Seed v0.7 low-rate follow-ups — 99k to 103k (2026-11-06)

## Result

Two additional 2,000-update uniform language continuations completed from the validation-best seed-73 lineage. Both resumed Forge's own weights, AdamW moments, tokenizer, and sample RNG. Learning rate was halved at each stage; the independent test range was also changed at each stage.

**Both stages improved validation and their newly reserved TinyStories test slice.** This extends the three-seed 95k-to-99k result as a repeatable text-prediction signal. Capability probes remain at chance.

| Stage | Updates and LR | Validation parent → candidate | Fresh test parent → candidate | Candidate SHA-256 |
|---|---|---:|---:|---|
| 99k → 101k | 2,000 at 0.00005 | 2.03671362 → 2.02064115 (−0.01607247) | 2.08334252 → 2.06656882 (−0.01677370, ~0.81%) | `94d87368faf8bad7fb34a43ca75448ddcd96bf2079dd731d4b5a0b808baf2eda` |
| 101k → 103k | 2,000 at 0.000025 | 2.02064115 → 2.01032551 (−0.01031564) | 1.99410557 → 1.98412743 (−0.00997814, ~0.50%) | `ec9a85b7a9d3787955d89d82f37bc2b494d96cc6ab5c106138f10285cdd2a868` |

Each run evaluated the same 2,048 evenly spaced 128-token validation windows every 500 updates. Validation improved at every check, so neither run hit its two-check early-stop limit. The independent test windows came from fresh 4 MiB byte ranges after the pinned TinyStories training prefix: the first follow-up used a 4 MiB offset and the second an 8 MiB offset. Exact train/validation duplicate stories were filtered. **Do not compare the test-loss numbers between the two ranges**; compare parent and candidate only within each row.

## Capability diagnostics

For both follow-ups, the existing 40-choice forced-A/B diagnostic scored **20/40** on both parent and candidate, with **0/20** mirrored pairs fully correct. The separate fresh 20-scenario probe also scored **20/40** for both parent and candidate, with **0/20** fully correct pairs. The probes are limited first-token A/B diagnostics. No human grading of generated text was done. These runs support a language-prediction improvement only, not a comprehension, reasoning, instruction-following, or Android capability claim.

## Provenance and artifacts

- The 99k parent was selected by validation (seed 73) from [matched replication run 42289055100](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42289055100); parent SHA-256 `b1ba557fc08b2a93c49857499f8d5a13c76eb7a5eb78a94254018876350631ad`.
- 99k → 101k: [run 42290966213](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42290966213); [artifact 13540316354](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42290966213/artifacts/13540316354); [draft research release](https://github.com/tpylman90-ctrl/Forge-Core/releases/tag/untagged-16e1720ce62d940ccdbb).
- 101k → 103k: [run 42291976445](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42291976445); [artifact 13539529431](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42291976445/artifacts/13539529431); [draft research release](https://github.com/tpylman90-ctrl/Forge-Core/releases/tag/untagged-84ea3f2e0d26dd9b46b9).
- Data source remained TinyStories revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, license CDLA-Sharing-1.0. Each run records its train, validation, new test, tokenizer, checkpoint, optimizer/RNG, and lineage hashes in the artifact.
- Both runs passed their training, evaluation, artifact, and draft-release steps; CI passed. All outputs remain research-only and separate from Android.

## Decision

The selected 103k checkpoint is the current strongest language-prediction research candidate in this lineage. The improvement continued with diminishing returns as the learning rate fell, while capability probes did not move. Do not automatically promote it. Further TinyStories-only continuation should pause until a task-focused curriculum or stronger capability evaluation is prepared; keep language, Blender, and Godot candidates isolated and compare them against their own controls.
