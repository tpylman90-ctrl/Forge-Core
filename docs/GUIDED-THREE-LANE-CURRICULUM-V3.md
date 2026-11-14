# Guided Three-Lane Curriculum v3

## Scope

This experiment has three separate paths: Forge Seed language, Blender 5.4, and Godot 4.7. Each path uses three seeds (43, 53, 73), its own candidate artifacts, validation, and held-out test report. No weights are merged, and no checkpoint is promoted to Android.

## Shared source lineage

All candidates begin from the exact selected Forge Seed v0.7 parent:
- Actions run: 42552572467
- Step: 107250
- Checkpoint SHA-256: 848553a81549c1269c360c5d07c57adde92a7816a0bc19b61ecc8028c93fee99
- TinyStories revision: f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
- Declared license: CDLA-Sharing-1.0
- Tokenizer SHA-256: 4ecd7803d4c369e80bda2e74c30cfb769b7e3a942f931df83383eec9f794f04c

The language lane resumes Forge's own optimizer. Blender and Godot each use a new, lane-local optimizer. Blender documentation is fetched once per workflow run from the pinned Blender 5.4 manual, with prose/code licensing rules and attribution recorded. Godot uses the pinned 4.7 manual and attribution. All seeds within one domain run consume the same source artifact and hashes.

## Small progressive runs

Each seed performs three sequential 500-update micro-stages on one isolated candidate:
1. Foundation: first 25% of the pinned training token sequence.
2. Expansion: first 60%.
3. Full mix: the complete pinned training split.

Validation runs every 100 updates. Each stage records its data-pool SHA-256 and validation curve. The final test split is used once after selection, never for tuning.

The language path trains only on TinyStories and selects by fixed TinyStories validation loss.

The Blender and Godot paths use a balanced batch: two domain windows plus two TinyStories replay windows. They select a checkpoint only when domain validation improves and shared TinyStories validation remains within 0.05 loss of the parent. Two consecutive guard breaches stop that seed. The exact limit is a conservative regression guard, not proof of language retention.

## What the reports mean

Reports include per-seed parent/candidate domain validation and test losses, the common language-retention delta, stage hashes, learning settings, checkpoint hashes, and stop reasons. Specialist task samples are ungraded. Documentation next-token loss can show text-pattern learning, but cannot establish that Forge can operate Blender or Godot. Human review and executable application tasks are separate future gates.

Action artifacts retain each seed candidate and source lineage for 90 days. The lane summary compares all three seeds, and each lane workflow creates a draft archive with all candidates, source hashes and report. Drafts remain research-only; no weights are merged or integrated into Android.
