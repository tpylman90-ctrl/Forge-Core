# Forge Seed dialogue bootstrap: 500-update CPU run

**Run:** [GitHub Actions run 44660738727](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/44660738727)  
**Result artifact:** retained in GitHub Actions through 2026-12-20  
**Source parent:** Forge Seed v0.7, run 42552572467, step 107250  
**Updates:** 500 per arm, two seeds (53 and 73)

| Seed | Correct-target | Shuffled control | Frozen parent | TinyStories loss change | Gate |
| ---: | ---: | ---: | ---: | ---: | :--- |
| 53 | 121/128 (0.9453) | 120/128 (0.9375) | 103/128 (0.8047) | +0.055316 | Fail |
| 73 | 123/128 (0.9609) | 119/128 (0.9297) | 103/128 (0.8047) | +0.055308 | Fail |

The candidates still outscored the frozen parent and narrowly beat their
shuffled-target controls. However, both failed the language-retention gate:
the maximum allowed validation-loss increase was +0.05. The 160-update
replications had passed with increases of +0.0162 to +0.0168 and scored
124/128 on both new seeds, so extending training to 500 did not improve this
probe and crossed the retention limit. Training loss also fell close to zero,
which is consistent with overfitting these small synthetic training examples.

**Disposition:** both 500-update checkpoints remain research-only artifacts;
neither was promoted or substituted for Forge's source weights. Keep the
workflow's normal default at 160 updates. A future run should test an
early-stopping or regularized schedule against the same frozen parent and
retention gate before spending more updates.
