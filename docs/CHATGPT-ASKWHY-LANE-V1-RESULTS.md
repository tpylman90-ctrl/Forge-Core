# ChatGPT Ask-Why lane v1 results

**Actions run:** [44669914051](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/44669914051)  
**Status:** All nine CPU branches and the summary job completed successfully. Each candidate, exact parent, direct lesson pack, Ask-Why lesson pack, and metrics are in the run's 90-day artifacts.

The three arms per lane used the same seed and pinned parent. Updates are optimizer steps attempted; specialist checkpoints were selected using the existing validation and retention logic. Positive lesson gain means lower held-out next-token loss.

| Lane | Arm | Updates | Direct lesson gain | Ask-Why lesson gain | Lane test: parent → selected | Shared-language Δ |
|---|---|---:|---:|---:|---:|---:|
| Language | Control | 500 | -0.00497 | +0.00336 | 1.99216 → 1.99175 | -0.00042 |
| Language | Direct ChatGPT | 500 | +1.36916 | +0.67828 | 1.99216 → 2.03279 | +0.04063 |
| Language | Ask-Why | 500 | +0.99025 | +1.59361 | 1.99216 → 2.03027 | +0.03811 |
| Blender | Control | 400 | +0.99794 | +0.80205 | 5.49547 → 4.48935 | +0.04654 |
| Blender | Direct ChatGPT | 400 | +1.54919 | +1.12403 | 5.49547 → 4.58758 | +0.04860 |
| Blender | Ask-Why | 300 | 0.00000 | 0.00000 | 5.49547 → 5.49547 | 0.00000 |
| Godot | Control | 400 | +1.03970 | +0.85180 | 5.77432 → 4.74896 | +0.04957 |
| Godot | Direct ChatGPT | 400 | +1.70369 | +1.16058 | 5.77432 → 4.66923 | +0.04281 |
| Godot | Ask-Why | 400 | +1.71832 | +1.86300 | 5.77432 → 4.59517 | +0.04543 |

## What changed

- **Language:** Ask-Why lowered the held-out Ask-Why lesson loss more than direct teaching did (1.594 vs 0.678). Direct teaching still had the lower loss on direct lessons (3.937 vs 4.316). Both trained language branches raised the TinyStories test loss by about 2%.
- **Godot:** Ask-Why led on held-out Ask-Why loss and had the best lane test loss of the three arms (4.595 vs 4.669 direct and 4.749 control). Shared-language validation rose by about 2.27%, under the existing 5% specialist guard.
- **Blender:** Ask-Why early-stopped and retained the parent checkpoint. Direct teaching lowered lesson-text loss, but its lane test was slightly worse than control (4.588 vs 4.489).

## Decision

The new step is wired into all three training lanes as a distinct, attributable arm:

**Forge asks why → ChatGPT explains in short steps and an analogy → Forge restates → applies → names a check.**

The one-seed result is a promising format-learning signal for Godot and a style-specific signal for language. It does not show autonomous question asking or verified tool competence: all added evaluations are held-out text prediction. Blender did not benefit in this pilot.

No candidate weights were promoted, merged, or installed. All nine branch artifacts are preserved with their parent and data for 90 days. Independent-seed replication and task-level evaluations should precede durable checkpoint archival.
