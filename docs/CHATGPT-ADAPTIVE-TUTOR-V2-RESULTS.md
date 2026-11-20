# ChatGPT adaptive tutor v2 results

**Training run:** [44673370274](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/44673370274)  
**Independent language replication:** [44674243069](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/44674243069)  
**Pinned parent:** Forge Seed, step 107,250; checkpoint SHA-256 `848553a81549c1269c360c5d07c57adde92a7816a0bc19b61ecc8028c93fee99`.

All jobs completed successfully. The four Blender/Godot jobs used distinct specialist seeds. In the first six-job run, language seeds 43 and 44 were identical because the language trainer restored the parent's saved sampling RNG. The independent replication workflow then reran language seed 44 with its own RNG stream while keeping the parent weights and optimizer fixed. The duplicate language-44 result is included below as a reproducibility check, not a separate replication.

The maximum budget was 500 updates. Specialist lanes selected the checkpoint that passed existing validation and shared-language retention gates; they stopped selecting after 100 updates. Positive lesson gain means lower held-out next-token loss.

| Lane | Sampling | Selected updates | Tutor-v2 gain | Direct gain | Ask-Why v1 gain | Lane test: parent → candidate | Shared-language Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| Blender | Seed 731 | 100 | +1.38897 | +1.36913 | +1.65258 | 5.49547 → 4.55590 | +0.04799 |
| Blender | Seed 732 | 100 | +1.29900 | +1.24485 | +1.51644 | 5.49547 → 4.61963 | +0.03763 |
| Godot | Seed 947 | 100 | +1.33949 | +1.58974 | +1.69374 | 5.77432 → 4.70331 | +0.04720 |
| Godot | Seed 948 | 100 | +1.24070 | +1.46780 | +1.56223 | 5.77432 → 4.72935 | +0.03359 |
| Language | Seed 43, parent RNG | 500 | +1.45233 | +0.97816 | +1.64017 | 1.99216 → 2.02905 | +0.03635 |
| Language | Seed 44, independent RNG | 500 | +1.47406 | +0.99784 | +1.64980 | 1.99216 → 2.02878 | +0.03660 |

## Interpretation

- **Blender and Godot:** Both seeds in each lane reduced the lane-test loss. Their shared-language validation changes stayed below the existing +0.05 absolute retention limit. This is a repeatable held-out text-prediction signal for these specialist branches, not proof that Blender or Godot tasks will work.
- **Language:** The independent seed reproduced the tutor-text gain, but TinyStories test loss increased by about 1.84% in both runs. This is a measurable tradeoff, so the language candidate is not a clean general-language improvement.
- **Teaching format:** Tutor-v2 contains diagnosis, retrieval, and transfer cases. These runs compared each candidate with its parent, while the earlier v1 study supplied separate control/direct/Ask-Why comparisons. They did not include a matched control and direct-v2 arm, so these results do not isolate how much of the gain comes specifically from the new tutoring sequence.
- **Limits:** Held-out evaluations are still next-token text loss. No executable Blender/Godot task rubric or live interactive ChatGPT-to-Forge session was included.

No candidate weights were promoted, merged across lanes, or installed. Checkpoints, parent data, lesson packs, and metrics are retained in the linked Actions artifacts for 90 days. The next evidence step is a matched v2 control/direct comparison plus task-level tests with observable answer keys.
