# Forge training results — 2026-11-20

This note records the completed Ask-Why CPU run and the Kaggle GPU results visible in the Forge Core Kaggle notebook. All model results remain research-only.

## ChatGPT Ask-Why overnight CPU run

**Actions run:** [44678755322](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/44678755322)  
**Parent:** Forge Seed step 107,250, checkpoint SHA-256 `848553a81549c1269c360c5d07c57adde92a7816a0bc19b61ecc8028c93fee99`.  
**Method:** tutor-v2 lessons ask Forge to attempt a task, ask why, use a causal breakdown, retrieve the principle, transfer it to a changed case, and check the result. Each lane uses a separate candidate, evaluation stream, and artifact.

| Lane | Seed | Updates attempted / selected | Training time | Tutor-v2 held-out loss, parent → candidate | Shared-language validation delta |
|---|---:|---:|---:|---:|---:|
| Language | 43 | 40,000 / 40,000 | 80m 50s | 5.42236 → 4.51662 (−0.90574) | +0.02174 |
| Language | 44 | 40,000 / 40,000 | 83m 20s | 5.42236 → 4.51662 (−0.90574) | +0.02174 |
| Blender | 731 | 400 / 100 | 92s | 5.29335 → 3.90438 (−1.38897) | +0.04799 |
| Godot | 947 | 400 / 100 | 49s | 5.32421 → 3.98472 (−1.33949) | +0.04720 |

The Blender and Godot jobs stopped after the retention checks rejected later checkpoints; their retained checkpoint is the 100-update candidate. Language stayed within the trainer's 0.05 absolute regression limit, but its shared-language loss increased.

**Replication caveat:** the language seed-44 job did not use independent sampling. Both language seeds followed the same restored parent RNG state and produced identical traces and metrics. Treat these as one language run, not two independent replications. Set `--independent-sampling` for any future seed replication.

All four jobs succeeded, retained parent/data/checkpoint artifacts for 90 days, and marked candidates ineligible for promotion. These measures are held-out next-token losses; they do not certify practical language, Blender, or Godot capability. The initial attempt failed before training because its update ceiling exceeded the trainer's 40,000-update limit; the corrected run completed.

## Kaggle GPU Ask-Why language candidate

The notebook recorded one separate language candidate from the same pinned parent:

- 32,182 updates attempted; checkpoint at 2,100 selected updates; 601 seconds of training.
- Tutor-v2 held-out loss: 5.506423 → 5.114061.
- Lane validation: 2.008943 → 2.008757.
- Independent test: 1.996549 → 1.996352.
- Retention gate passed.

The lane and independent-test changes are very small. No practical task evaluation was reported, so this remains a research result. No evidence of a GitHub release or durable GitHub artifact export for this Kaggle candidate was found.

## Kaggle GSO v6 visual pilot

The pilot used 135 Google Scanned Objects assets marked CC BY 4.0, split by object ID into 108 training and 27 held-out assets (five 64×64 thumbnails per asset). A 25,763-parameter CNN predicted normalized mesh axis-aligned bounds from rendered thumbnails. Three 18-epoch runs used a Tesla T4.

| Method | Held-out MAE |
|---|---:|
| Training-set median baseline | 0.3065 |
| CNN, seed 43 | 0.3098 |
| CNN, seed 53 | 0.2884 |
| CNN, seed 73 | 0.3007 |

Two of three seeds beat the median baseline; the gain is small and seed-sensitive. This is not full geometry reconstruction and does not establish arbitrary photo-to-3D capability. The notebook reports that no Forge weights were changed or promoted. Its full JSON summary was written to Kaggle's `/kaggle/working/gso_v6_pilot_summary.json`; no durable GitHub export for that file or the CNN weights was found.

## Decision

Keep the Ask-Why and visual candidates separate from Forge's retained weights. The clearest gain is learning to predict the tutor-v2 response format; general language retention did not improve consistently, and the visual result is too small and seed-sensitive to promote. The next training run should fix the independent language sampler and use held-out task scoring before any promotion decision.
