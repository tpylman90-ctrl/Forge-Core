# Overnight Blender and Godot scene-spec SFT — 2026-11-12

Run the isolated eight-arm CPU experiment defined in `.github/workflows/scene-spec-sft-overnight.yml`.

- Four independent seeds per lane, from the exact verified specialist parents archived in run 42541109934.
- Original, deterministic CC0 examples; hash-addressed train/validation/test splits with the fixed engine probe held out.
- Answer-only loss with the prompt masked; no external model weights or teacher outputs.
- Up to 15,000 updates per seed, early stopping on validation answer loss, maximum 180 minutes per job.
- Blender and Godot lanes never share optimizer state or task data; language weights are not loaded.
- Compare parent and selected candidate with held-out exact fields, answer loss, and the same headless engine harness.
- Preserve per-seed checkpoint lineage, optimizer/RNG state, data hashes, reports, and all raw engine outputs.
- Research-only; no automatic promotion or Android integration.
