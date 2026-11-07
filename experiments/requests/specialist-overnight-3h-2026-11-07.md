# Overnight Forge specialist trials

Run one three-hour CPU trial for each of the Blender and Godot specialist lanes concurrently.

## Lineage and isolation
- Start both trials from the exact verified Forge Seed v0.7 95,000-step parent: run 42143390701, SHA-256 a7bcdb996e440f44d7496b887b54d19324927b4296fe9ed2f0f0293c88c63a0a.
- Use the pinned lane-specific documentation and lesson data, with per-page and split hashes, license records, and attribution retained.
- Keep Blender and Godot weights, optimizers, data, and artifacts separate. Do not alter or merge into the language lane.
- Save the best validation checkpoint and complete lineage/evaluation artifacts.

## Evaluation limits
The prior one-hour trials improved fixed documentation token loss: Blender 5.6520 to 3.5304 validation and 5.5796 to 3.8343 test; Godot 5.5651 to 4.2496 validation and 5.8475 to 4.3192 test. Those results measure prediction on small held-out documentation splits only. Task samples remain ungraded; do not claim tool competence or promote either candidate to Android.

This request starts .github/workflows/specialist-overnight-3h.yml.