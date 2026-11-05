# Forge Copy v7 — verified length-transfer result (November 5, 2026)

**Run:** https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42042371354

**Artifact:** https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42042371354/artifacts/13422941017

**CI:** https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42042371443

The bounded two-seed original-neural experiment completed. Both arms
used the same decoder architecture, starting weights within each
seed, optimizer, 500 updates, and 8 strings/update. Short-only
training used 3–8 characters; the curriculum introduced 9–12
characters after an initial short stage. The archived original
Forge Seed language checkpoint was verified but **not modified or
loaded into the standalone copier**.

| Seed | Arm | Novel short exact /256 | Novel long exact /240 | Long pairs both exact /120 |
| --- | --- | ---: | ---: | ---: |
| 13741 | Short only | 252 | 15 | 6 |
| 13741 | Length curriculum | **254** | **194** | **96** |
| 13742 | Short only | 252 | 37 | 18 |
| 13742 | Length curriculum | **254** | **204** | **100** |
| **Combined** | Short only | **504/512** | **52/480** | **24/240** |
| **Combined** | Length curriculum | **508/512** | **398/480** | **196/240** |

**Interpretation:** length exposure substantially improved
previously unseen 9–12-character copying without measurable short
accuracy collapse in this two-seed procedural study. Long exact
copying increased from 10.8% to 82.9%, while short accuracy stayed
above 98%. The change also altered which source strings were
sampled, so it is not a same-example ablation.

This is **not preschool reading, grammar, semantic-role understanding,
or a demonstration of the original Forge Seed language model's
abilities**. It is an independent, randomly initialized source-copy
neural accessory using direct alignment supervision and procedural
data. No teacher pretrained weights, private data, Android observer
changes or checkpoint promotion. Future work should test new
length distributions, unfamiliar input formatting, and integration
with actual language questions without leaking gold source positions
at inference.
