# Best-lane continuation results — 2026-11-08

## Run and lineage

GitHub Actions run [42480865448](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42480865448) completed all three training and evaluation jobs successfully. The completed checkpoint and data bundles are preserved in the [draft research release](https://github.com/tpylman90-ctrl/Forge-Core/releases/tag/untagged-4cab3ca0c18f097d8767).

Each lane resumed its own strongest checkpoint from run 42445321887, restored optimizer moments and sampler RNG, and remained a separate model/data branch. The language lane continued Forge Seed v0.7 from step 106,000; Blender and Godot resumed their isolated 1,500-update specialists. Training used no Qwen or ChatGPT weights/logits. Blender and Godot used the original CC0 task/answer/why cards and pinned official-manual data. Source manifests, data hashes, tokenizer, parent, optimizer, selected checkpoint, last checkpoint, and evaluation outputs are in the archived bundles.

## Results

| Lane | Updates completed | Validation parent → selected | Independent test parent → selected | Selected branch |
|---|---:|---:|---:|---|
| Language | 1,000 | 2.00455089 → 2.00338817 | 1.96600301 → 1.96474480 | New candidate at step 107,000 |
| Blender | 500 (early stop) | 3.53043514 → 3.53043514 | 3.83430475 → 3.83430475 | Retained parent |
| Godot | 500 (early stop) | 4.20090926 → 4.20090926 | 4.27263499 → 4.27263499 | Retained parent |

The language loss decreases are small: 0.00116272 on validation and 0.00125822 on the fixed independent test. This supports a modest text-prediction gain on these fixed samples only.

Blender and Godot validation losses worsened at both early-stop evaluations, so the selectors kept each parent. Their last trained states remain archived for research comparison; they are not the selected checkpoints. The fresh manual-document losses and task scores were unchanged because the selected checkpoint was the parent: Blender 3.77082615 and 0/48; Godot 3.81662841 and 5/60.

## Language capability check

The existing forced-choice probe changed from 20/40 to 19/40; both versions solved 0/20 mirrored pairs in both orientations. The candidate did not meet the configured threshold. No human sentence/story grading was performed. It remains uncertified.

## Checkpoint hashes

- Language selected candidate: `5f050ad1ff11d8d15603e27c90ce214464c4f4899fda7e7984cbc9d063c87337`
- Language parent: `b038869062c3b051768b10b1f146aeb8fdde7419cfdd2e8ebdf54acfbc2ea42b`
- Blender selected parent: `672db59bfc44d026f26e38d781c84dd24f92882c65b888639ec66a790002ed68`
- Godot selected parent: `de7de5052f0a56d821ea3995f1f2e5fbb80d10750e4fb3a564f3c6c4a2a0afdf`

All results are research-only. No checkpoint was promoted to Android or presented as a capability upgrade.
