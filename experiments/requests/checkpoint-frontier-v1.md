# Checkpoint frontier v1 run request

Run the checkpoint-frontier workflow on main.

This is a controlled research continuation, not a release or app promotion:
- Parent checkpoints are the exact per-seed candidates from the completed guided three-lane v3 run on 2026-11-14.
- Language lane compares inherited learning-rate multipliers 0.5x, 1.0x, and 2.0x across seeds 43, 53, and 73.
- Blender 5.4 and Godot 4.7 each compare nominal TinyStories replay fractions 25%, 50%, and 75% across the same three seeds.
- Every branch runs 1,500 progressive updates with validation every 100; each specialist branch retains the existing +0.05 shared-language guard.
- Keep source, parent and candidate hashes, metrics, and parent checkpoint copies in the run artifacts. Do not merge candidates, promote them to Android, or claim tool competence from token loss.
