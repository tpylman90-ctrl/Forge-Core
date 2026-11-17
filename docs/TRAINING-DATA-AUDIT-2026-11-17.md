# Forge training data audit and next-run decisions — 2026-11-17

## Scope

This review compared the current training/evaluation sources with the recorded outcomes for core language, Blender, Godot, Qwen distillation, and the image-to-procedural-object branch. The most recent workflow and artifacts were also checked. Personal memory is not training data, and the specialist branches remain isolated.

## Findings and decisions

### Core language

Forge's current language lineage uses the pinned TinyStories revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` (CDLA-Sharing-1.0) with tokenizer SHA-256 `4ecd7803d4c369e80bda2e74c30cfb769b7e3a942f931df83383eec9f794f04c`. It provides reproducibility, but the corpus is narrow: repeated continuation produced only very small text-loss changes, and the v4 language arm selected its unchanged parent for all three seeds.

The earlier Qwen3 distillation study used 192 training and 32 held-out teacher examples. A 10% teacher mix improved loss on teacher-style samples but worsened TinyStories validation and independent-test loss; the 1% replication did not give a repeatable core-language improvement. Concentrated high-loss replay and source-oracle correction traces also regressed core validation loss by about 15% in the matched study. These results do not support repeating those mixtures as core training.

**Decision:** defer further core-language updates until a broader, license-reviewed corpus and a new larger held-out evaluation are ready. Preserve the verified language parent and evaluate new data before investing more steps. The existing 256-window measurements must not be compared directly to the 2,048-window or 4,096-window evaluations.

### Blender and Godot

The specialist branches use pinned official manual text with source attribution and 50% TinyStories replay. V3 improved documentation held-out loss in all three seeds for both lanes. V4 again improved all Blender and Godot seeds, while language remained within its +0.05 retention guard. Because these are text-prediction metrics, they do not show that the model can complete real application tasks.

A follow-up was run from the matching v4 candidate for each seed. It resumed optimizer and sampler state, halved the learning rate to 5e-6, used balanced 50% TinyStories replay, trained three progressive 500-update stages, and checked validation every 100 updates. All six candidates completed 1,500 updates, improved held-out domain test loss versus their own v4 parent, and passed the core-language guard.

| Lane | Mean test loss, v4 parent → v5 candidate | Relative change | Mean TinyStories validation delta |
|---|---:|---:|---:|
| Blender | 3.9130 → 3.8797 | -0.85% | +0.0043 |
| Godot | 4.3250 → 4.3088 | -0.37% | +0.0049 |

The largest individual core-language delta was +0.0055, under the configured +0.05 guard. This is a positive, repeated next-token result on these pinned documentation splits, not proof of Blender/Godot competence. Human task grading and executable application tasks remain separate gates.

- Workflow: [run 44006869302](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/44006869302)
- Draft archive: [v5 candidate/source/checkpoint bundle](https://github.com/tpylman90-ctrl/Forge-Core/releases/tag/forge-guided-followup-v5-44006869302)
- v4 parent and method comparison: [v4 results](GUIDED-LANE-FOLLOWUP-V4-RESULTS-2026-11-16.md)

### Image and shape branch

The visual branch starts from fresh random weights and does not load Forge Seed language weights. Its fixed CC0-1.0 data has 960 procedural objects across six categories and four rendered views per object. V2's mask-aware geometry loss improved mean proxy silhouette IoU from 0.380 to 0.475 across three fresh seeds. That is a useful synthetic gain, but IoU is still below 0.5. The prompts expose category and approximate dimensions, and the held-out objects come from the same small procedural vocabulary; category accuracy is therefore not meaningful evidence of open-world recognition.

**Decision:** before another visual training run, improve the data and evaluation: hold out whole geometry/template families, vary camera/light/background/render conditions, include more object/component families, make prompts less answer-revealing, and retain geometry metrics plus multi-view silhouette scoring. Add real or independently authored references only after license review. Keep this as its own model branch.

## Next work

1. Keep the six v5 Blender/Godot checkpoints archived per seed. Do not merge lanes or promote candidates to Android based only on documentation loss.
2. Add held-out Blender and Godot tasks that require valid executable scripts or scene/property changes, with exact assertions and human review for quality.
3. Build and hash a broader, license-reviewed language corpus plus a larger independent test set; then compare low-rate uniform continuation against a no-update parent across multiple seeds.
4. Improve visual data family diversity and evaluate on unseen shape families before spending more updates.
5. Keep source revisions, licenses, corpus hashes, checkpoint parents, optimizer/RNG state, selected-step bookkeeping, and independent evaluations in every training archive.

All candidate weights discussed here remain research-only. No branch was merged, Android-integrated, or described as a capability upgrade.
