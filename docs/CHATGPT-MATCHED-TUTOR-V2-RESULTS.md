# Matched ChatGPT tutor-v2 comparison

**Actions run:** [44676106369](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/44676106369)  
**Status:** All nine training jobs and the summary job completed successfully.  
**Parent:** Pinned Forge Seed step 107,250, checkpoint SHA-256 `848553a81549c1269c360c5d07c57adde92a7816a0bc19b61ecc8028c93fee99`.

The study compared three arms per lane using the same parent, seed, trainer, 500-update budget, and 100-update evaluation cadence:

- **Control:** original lane stream only.
- **Direct-v2:** a worked explanation, procedure, principle, and same-case check.
- **Tutor-v2:** Forge's attempt and Ask-Why prompt, misconception diagnosis, short explanation and analogy, faded retrieval, a changed transfer case, and an observable check.

Positive lesson gain means lower held-out next-token loss. The lane test is also text prediction on the existing lane split, not an executable task test.

| Lane | Arm | Selected updates | Direct-v2 gain | Tutor-v2 gain | Lane test: parent → candidate | Shared-language Δ |
|---|---|---:|---:|---:|---:|---:|
| Language | Control | 500 | +0.01441 | -0.00300 | 1.99216 → 1.99175 | +0.00008 |
| Language | Direct-v2 | 500 | +2.11295 | +0.64668 | 1.99216 → 2.03631 | +0.04321 |
| Language | Tutor-v2 | 500 | +1.47126 | +1.45233 | 1.99216 → 2.02905 | +0.03635 |
| Blender | Control | 100 | +0.95673 | +0.64059 | 5.49547 → 4.48935 | +0.04654 |
| Blender | Direct-v2 | 100 | +1.62760 | +0.82315 | 5.49547 → 4.56952 | +0.04780 |
| Blender | Tutor-v2 | 100 | +1.33744 | +1.38897 | 5.49547 → 4.55590 | +0.04799 |
| Godot | Control | 100 | +1.21659 | +0.66189 | 5.77432 → 4.74896 | +0.04957 |
| Godot | Direct-v2 | 100 | +1.73883 | +0.77659 | 5.77432 → 4.69340 | +0.04481 |
| Godot | Tutor-v2 | 100 | +1.36150 | +1.33949 | 5.77432 → 4.70331 | +0.04720 |

## Findings

- **Teaching-format prediction:** Tutor-v2 had the largest tutor-v2 held-out gain in every lane. Direct-v2 had the largest direct-v2 held-out gain in every lane. This is strong evidence that each branch learned the format it practiced.
- **Lane-text retention:** Control had the lowest lane-test loss in language and Blender. Direct-v2 had the lowest Godot lane-test loss, narrowly ahead of tutor-v2. The tutor sequence did not win the existing lane test in any lane.
- **Language tradeoff:** Direct-v2 and tutor-v2 both raised TinyStories test loss; control remained essentially flat. The tutor branch's rise was about 1.85%. Do not treat this as general language improvement.
- **Specialist retention:** Blender and Godot shared-language changes stayed under the existing +0.05 absolute gate, though near its limit for some arms.

## Decision

No candidate is promoted, merged, or installed. All candidates, exact parents, lesson packs, and metrics are in the run's 90-day artifacts.

The Ask-Why dialogue is worth keeping as a research teaching format because it improves prediction of unseen examples in that format. It has not yet shown an advantage on practical Blender/Godot tasks. The next efficacy gate should grade unseen tasks with concrete answer keys or safe tool execution, then replicate promising arms across independent seeds.
