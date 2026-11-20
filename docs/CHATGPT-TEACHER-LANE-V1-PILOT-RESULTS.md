# ChatGPT teacher lane v1 pilot results

**Run:** [Forge Core Actions run 44664573454](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/44664573454)  
**Result:** All six CPU jobs and the summary job completed successfully. Candidate checkpoints, parent checkpoints, teacher data, and metrics are retained together in the run's 90-day artifacts.

Each lane used its fixed seed and the same pinned Forge Seed parent. Language ran 500 updates; Blender and Godot stopped after 400 updates under validation-based early stopping. Lower loss is better.

| Lane | Arm | Held-out teacher loss: parent → candidate | Lane test loss: parent → candidate | Shared-language validation |
|---|---|---:|---:|---:|
| Language | Control | 5.30636 → 5.31133 | 1.99216 → 1.99175 | — |
| Language | ChatGPT | 5.30636 → 3.93721 | 1.99216 → 2.03279 | test loss increased about 2.0% |
| Blender | Control | 5.45723 → 4.45928 | 5.49547 → 4.48935 | 2.00255 → 2.04909 (+2.32%) |
| Blender | ChatGPT | 5.45723 → 3.90803 | 5.49547 → 4.58758 | 2.00255 → 2.05115 (+2.43%) |
| Godot | Control | 6.21418 → 5.17449 | 5.77432 → 4.74896 | 2.00255 → 2.05212 (+2.48%) |
| Godot | ChatGPT | 6.21418 → 4.51049 | 5.77432 → 4.66923 | 2.00255 → 2.04536 (+2.14%) |

## Readout

The ChatGPT lesson arm reduced held-out teacher-text loss in all three lanes compared with the matched control: language 3.937 vs 5.311, Blender 3.908 vs 4.459, and Godot 4.510 vs 5.174.

That is evidence the model learned to predict this lesson style. It is not yet proof of better task performance. On the existing lane text test, ChatGPT was slightly worse than control for Blender (4.588 vs 4.489), slightly better for Godot (4.669 vs 4.749), and increased language test loss by about 2.0%. Specialist shared-language validation drift remained below the 5% guard in both Blender and Godot branches.

## Decision

Keep this as a promising, isolated pilot. No weights were promoted, merged, or installed. The three checkpoint pairs remain in the Actions artifacts for 90 days; the lesson packs and study implementation are committed permanently. Before durable checkpoint archival, repeat the promising language and Godot findings with independent seeds and evaluate held-out task answers. Blender needs task-level evidence because its lane test was weaker than control.
