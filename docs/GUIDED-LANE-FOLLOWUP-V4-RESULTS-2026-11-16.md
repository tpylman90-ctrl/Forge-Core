# Forge guided lane follow-up v4 — 2026-11-16 UTC

Status: research only. No lane was merged, promoted, or integrated into Android.

Run: https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/43804101939
Workflow commit: f17b42076b10905a9ee3a203a489099995181600
Trainer bookkeeping fix committed afterward: 413729b0ace2ae4c011fab816e68e7c10656d16f

Each of three seeds continued from its own v3 candidate. Each attempted 1,500 updates in three growing data stages, with validation every 100 updates. Blender and Godot used 50% TinyStories replay and the existing +0.05 core-language-loss guard.

| Lane | Seed | Domain test loss, parent → candidate | Core language Δ | Guard | Candidate checkpoint SHA-256 |
|---|---:|---:|---:|:---:|---|
| Blender | 43 | 4.1160 → 3.9023 | +0.0170 | pass | d1506871f4a01f22938288596a88a8f8873a49a47a39db42386aeeb31c883c3d |
| Blender | 53 | 4.1293 → 3.9186 | +0.0161 | pass | 3286fde866d274ee6905c6c02cac0983a81f30d0e5bcc60e77ae4596451ad802 |
| Blender | 73 | 4.1228 → 3.9182 | +0.0176 | pass | 9c3b5acaec29311457b1ac303a526903bb08f1b9d0f0c20aef4fa18dddc1d593 |
| Godot | 43 | 4.4422 → 4.3170 | +0.0162 | pass | 5a32b561768dcb1cc4816400ac4309b7ce16965e42b22b40c60d5982a683891c |
| Godot | 53 | 4.4485 → 4.3144 | +0.0160 | pass | a60ff804a4b5ffdc7c0fed6873bddb42ffca41a00a1cfc2e4a6ccf4b3306d1ba |
| Godot | 73 | 4.4638 → 4.3438 | +0.0174 | pass | 2c42e3a4fbbd5d7c4d8736b0cbbdc033c8f1c64ba9222044409eec1dae4a48f8 |
| Language | 43 | 1.9915 → 1.9915 | +0.0000 | pass | b93775d6b8b464ac94aa4794004003d961ea5207d37b8ce2737b9247cef331bc |
| Language | 53 | 1.9905 → 1.9905 | +0.0000 | pass | e45b57a5e6a93740b715ab7ccc7e074d7b6b98b49d9a08a726c82917e0128a9e |
| Language | 73 | 1.9913 → 1.9913 | +0.0000 | pass | 0812d6152dd6f7dd8cbc7c468863d986dbd8e7977f41acbde0202d0b61e211e2 |

Blender test loss fell by about 5.2% on average, and Godot by about 2.8%, consistently across the three seeds. This is a repeatable gain on pinned held-out documentation text. It does not measure successful Blender/Godot project tasks or establish practical competence. The core-language loss rose by 0.0160–0.0176 for specialist lanes, remaining within the configured guard.

The language arm did not improve held-out loss in any seed. The trainer selected the parent weights in each case. These artifacts were produced before the checkpoint-step bookkeeping fix; their reported step can reflect attempted updates even when the selected weights are unchanged. Do not use those three language artifacts as advanced language parents. The fix now records the selected checkpoint step, cumulative lane updates, and resumes optimizer/sampler state for an existing isolated lane branch.

All parent checkpoints, source data, manifests, curves, metrics, and candidate checkpoints are in the run artifacts and draft release:
https://github.com/tpylman90-ctrl/Forge-Core/releases/tag/untagged-6388dcf87efad18c4f05
