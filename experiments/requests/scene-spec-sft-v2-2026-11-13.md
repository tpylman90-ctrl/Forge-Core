# Transfer-focused Blender and Godot SFT v2 — 2026-11-13

Execute six independent answer-masked CPU SFT runs from the v2 workflow.

- Three seeds per lane with a maximum of 5,000 updates and validation early stopping.
- Blender starts from the best archived v1 Blender candidate (seed 741); the original verified Blender parent remains the baseline.
- Godot starts from the verified original 1,500-update parent; the v1 Godot candidate remains an evaluation control because its varied held-out exact-task score was 0/24.
- Use deterministic original CC0 task data across cube, UV sphere, and cylinder families for Blender; SphereMesh, BoxMesh, and CapsuleMesh families for Godot.
- Split by task composition. Keep prompt forms disjoint, with two unseen held-out prompt forms per family. Include fixed probes in held-out data.
- Evaluate 102 stratified held-out task prompts per run, plus multi-family headless engine checks. Compare original parent, v1 candidate, training start, and selected checkpoint on the same held-out set.
- Preserve exact checkpoint lineage, data and split hashes, optimizer/RNG state, metrics, and raw engine output. No teacher-model weights, language checkpoint, promotion, or Android integration.


Restart after manifest syntax fix.

Restart after held-out prompt selection fix.

Restart after adding the fixed Blender probe value.

Restart after fixing the generated Godot project string in the multi-family evaluator.
