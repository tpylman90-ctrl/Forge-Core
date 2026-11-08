# Controlled night growth batch — results (2026-11-07)

The 12-trial batch completed successfully in [GitHub Actions run 42445321887](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42445321887). CI passed. Four fixed-seed sampling trials ran in each lane. They all start from shared parent weights within their lane; the seeds vary sample order and are not independent initializations.

## Summary

- **Language:** 3 of 4 selected checkpoints improved both validation and the fresh 16 MiB-offset TinyStories test slice. The fourth ran 500 updates, did not beat validation, and selected the unchanged parent.
- **Blender:** all four 1,500-update candidates improved the lane's fixed documentation validation and test losses.
- **Godot:** all four candidates improved the lane's fixed documentation validation and test losses. Seed 949 stopped early after 1,250 updates.
- These results show small-sample text-prediction gains on the measured data. The tool-lane task samples were not human graded, and the fresh task probe was not rerun for these 8 candidates.

## Language trials

Parent: step 105,000, SHA-256 `379cd45eb95e74c72a91aaa5dd68f5efb266c4626198279d2fb021a019102bf4`.
Each run used the pinned TinyStories revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` (CDLA-Sharing-1.0), 1,000-update cap, learning rate 0.0000125, optimizer moments resumed, validation every 250 updates, and a fresh 16 MiB-offset test range.

| Matrix seed | Sample-order policy | Updates | Validation parent → selected | Fresh test parent → selected | Selected checkpoint SHA-256 |
|---:|---|---:|---:|---:|---|
| 43 | Resume parent RNG (stored parent sampler label 73) | 1,000 | 2.00642415 → 2.00488551 | 2.05114278 → 2.04951270 | `a2eb96a84f6241ccae5341dcac5b09e2758af1c92f21c4c69a01b0d1a524bb92` |
| 73 | Use generator seed 74 | 500 | 2.00642436 → 2.00642436 | 2.05114297 → 2.05114297 | Parent retained: `379cd45eb95e74c72a91aaa5dd68f5efb266c4626198279d2fb021a019102bf4` |
| 101 | Use generator seed 102 | 1,000 | 2.00642436 → 2.00474042 | 2.05114297 → 2.04965571 | `442d44956410421c2bfb5e2bede9b266ca295898fd3c6fb2e3b253fb1f5ff6ab` |
| 131 | Use generator seed 132 | 1,000 | 2.00642415 → 2.00455068 | 2.05114278 → 2.04926488 | `b038869062c3b051768b10b1f146aeb8fdde7419cfdd2e8ebdf54acfbc2ea42b` |

The internal metric field records the effective parent sampler label for matrix seed 43, and records the matrix seed (rather than the incremented generator seed) for the other runs. The table uses the matrix seed and spells out the actual RNG policy to make the four runs unambiguous.

## Blender trials

All candidates were trained for 1,500 updates at learning rate 0.00005 from the verified Forge 95k parent, SHA-256 `a7bcdb996e440f44d7496b887b54d19324927b4296fe9ed2f0f0293c88c63a0a`. They used the same pinned Blender 5.4 documentation split and the original CC0 task/answer/why lesson cards.

| Seed | Validation parent → selected | Test parent → candidate | Candidate checkpoint SHA-256 |
|---:|---:|---:|---|
| 731 | 5.65203571 → 3.53043514 | 5.57955902 → 3.83430475 | `672db59bfc44d026f26e38d781c84dd24f92882c65b888639ec66a790002ed68` |
| 732 | 5.65203566 → 3.54427271 | 5.57955898 → 3.85327735 | `680e3659b09e87e1a3d66fc2aa9233729ffd5a6c6fa0ff5b878507fd56c233b6` |
| 733 | 5.65203569 → 3.54508238 | 5.57955897 → 3.85854009 | `efc967a3ce52ce7bef3e880521d2ddeb9b493a61001125d9bd25239e4782ccf5` |
| 734 | 5.65203571 → 3.54588448 | 5.57955902 → 3.84937260 | `ad885dec817241cea3f96c5ce40e270fed76fe400d1177ebe46a2640f3b0778a` |

Mean candidate loss: validation 3.54142; test 3.84887.

## Godot trials

All candidates were trained at learning rate 0.00005 from the same verified Forge 95k parent. They shared one pinned Godot 4.7 documentation split and the original CC0 task/answer/why lesson cards.

| Seed | Updates | Validation parent → selected | Test parent → candidate | Candidate checkpoint SHA-256 |
|---:|---:|---:|---:|---|
| 947 | 1,500 | 5.56509919 → 4.20966174 | 5.84745647 → 4.25065584 | `a1edf448b6dcef65b51306b7eac164b598479d0f0886fe4cd1921d512047ff22` |
| 948 | 1,500 | 5.56509915 → 4.21884631 | 5.84745646 → 4.28681309 | `538915a7bf818cee757910a0d38032621b55a1c57812b5df597a34ffea9652bd` |
| 949 | 1,250 | 5.56509915 → 4.21877808 | 5.84745646 → 4.26240186 | `212c9e94d57e09efeedb72864dabdafaf97d351eefedd5f849f2e570263ecc3a` |
| 950 | 1,500 | 5.56509915 → 4.20090926 | 5.84745646 → 4.27263499 | `de7de5052f0a56d821ea3995f1f2e5fbb80d10750e4fb3a564f3c6c4a2a0afdf` |

Mean candidate loss: validation 4.21205; test 4.26813.

## Teacher and interpretation notes

GitHub Actions did not query this chat model or receive its logits/weights. Blender and Godot used existing original Forge lesson cards that pair tasks with code answers and explanations of why; their source manifest marks these cards CC0-1.0. Language used uniform TinyStories continuation. We did not repeat Qwen mixing because its earlier low-dose and 10% trials worsened independent TinyStories loss.

The Blender and Godot gains are documentation next-token loss reductions on one fixed split per lane. They do not show that Forge can reliably execute Blender or Godot tasks. No specialist checkpoint was merged into language weights or promoted to Android. Full checkpoints, optimizer states, manifests, and reports are retained in the run artifacts and draft release.
