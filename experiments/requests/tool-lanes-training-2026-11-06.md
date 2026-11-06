# Isolated Blender and Godot specialist run request

Approved run scope: train the Blender and Godot specialist lanes for up to 3,600 seconds each, concurrently, on CPU GitHub Actions.

- Start both lanes from the exact verified Forge Seed v0.7 95,000-step parent (run 42143390701; checkpoint SHA-256 a7bcdb996e440f44d7496b887b54d19324927b4296fe9ed2f0f0293c88c63a0a).
- Keep each specialist optimizer, candidate weights, data, and artifacts separate from the language lane and from the other specialist lane.
- Use the pinned Blender 5.4 Manual prose (CC BY-SA 4.0, code examples excluded) and Godot 4.7 documentation (CC BY 3.0), with the original Forge lesson cards (CC0-1.0); preserve source and split hashes and attribution.
- Record held-out documentation loss and deterministic task samples. Generated samples remain ungraded, so report no competence certification.
- Keep outputs research-only. Do not merge into the language checkpoint, replace the frozen parent, or integrate into Android.

This request file triggers .github/workflows/tool-lanes-training.yml.