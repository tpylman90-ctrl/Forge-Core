# Continue the three isolated Forge learning lanes from their verified v3 candidates

Run the v4 follow-up workflow for three fixed seeds in each lane.

Parents are the exact successful v3 artifacts:
- Language: run 43553138349, seeds 43, 53, and 73.
- Blender: run 43553140852, seeds 43, 53, and 73.
- Godot: run 43553141871, seeds 43, 53, and 73.

Each run must verify the parent checkpoint SHA, seed, lane, tokenizer, pinned TinyStories revision and license before training. Continue each lane independently for three progressive stages of 500 updates each, with held-out checks every 100 updates. Blender and Godot retain balanced 50% TinyStories replay and the core-language regression guard. Preserve parent checkpoints, optimizer state when applicable, source splits, manifests, training curves, candidate hashes, and evaluation results in workflow artifacts and a draft release.

Do not merge lane weights, promote to Android, or claim practical competence from token-loss changes alone. Compare each candidate with its own exact parent and report all nine seeds.
