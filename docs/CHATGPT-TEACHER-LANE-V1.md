# ChatGPT teacher lane v1

This study adds an explicitly attributed ChatGPT-authored teaching channel beside Forge Core's existing rule-verified lessons and public documentation/language-data lanes.

## Pilot

The CPU workflow trains six branches: control and ChatGPT-teacher arms for language, Blender, and Godot. Each starts from the same pinned 107,250-step Forge Seed parent. The ChatGPT arm mixes a CC0 lesson pack into its lane training stream; the control arm uses the existing stream. Both arms use the same lane trainer, seed, evaluation splits, and retention checks.

The held-out teacher score is next-token loss on examples reserved from training. Lower loss shows better prediction of that text. It does not demonstrate reliable conversation, Blender operation, Godot development, or broad reasoning. This first run uses one seed per lane and is a pilot.

## Storage and overlap

The exact parent, candidate, teacher pack, and metrics are kept together in 90-day Actions artifacts. No candidate is automatically promoted or merged. A promising pilot must repeat across independent seeds and improve on a separate task-level evaluation before durable checkpoint archival.

The packs are separately versioned so their examples can later be reviewed into a shared, provenance-tracked curriculum. This allows the learning channels to overlap through reviewed training data while preserving which channel taught each example. Checkpoints remain lane-specific; weight averaging or cross-lane consolidation requires a separate transfer experiment and regression gates.

The packs contain original synthetic examples, no private user conversation, and are released under CC0-1.0.
