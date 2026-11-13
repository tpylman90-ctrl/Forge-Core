# Godot exact-field SFT v3 training request

Run the three controlled CPU fine-tunes defined in `.github/workflows/scene-spec-godot-fields-v3.yml` with seeds 871, 872, and 873. Compare each selected candidate against the fixed original, v1, and v2 checkpoints using the same held-out split and pinned headless Godot validation. Preserve all metrics and candidate checkpoints as research-only artifacts. Do not promote weights or integrate them into Android.

Preflight follow-up: normalize every source wording variant that requests an explanation before training; keep the same data split and seeds.

Restore follow-up: when an archive contains duplicate copies with the same verified checkpoint SHA, select the source copy deterministically and continue with the same hash check.
