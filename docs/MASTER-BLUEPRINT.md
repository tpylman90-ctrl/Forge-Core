# Forge Core — Master Blueprint v2

**North star:** empower people without formal education, expensive tooling or specialized hardware to create professional 3D assets from their ideas. Forge is a specialist system, not a general assistant.

## Engineering model

Prefer deterministic and mathematically verifiable operations where possible. Use learned components for difficult perception or reconstruction tasks only when they outperform appropriate held-out baselines. Keep the canonical asset representation independent of any neural architecture. Start with Blender; preserve a stable operation interface for an eventual native engine.

## Release gates

A production candidate must include editable geometry, correct scale, validated mesh and materials, texture packaging, reproducible preview renders, and a report of limitations. A successful script, lower training loss, or silhouette metric alone is insufficient. Use original-versus-edited comparisons, held-out asset families, multiple seeds where applicable and measured time/resource budgets.

## Training and licensing

Use independently sourced, appropriately licensed datasets and explicit attribution; preserve data hashes, split manifests, source model license and checkpoint lineage. Keep CPU, GPU and experimental candidates separate. No general language or assistant training in the active project. Do not use proprietary assistant outputs as training material without applicable permission and review. No automatic weight promotion or publication.

## Delivery order

Follow [roadmap](ROADMAP.md), [architecture](ARCHITECTURE.md) and [lane rules](OPEN-3D-TRAINING.md). The previous blueprint remains available in the [archive branch](https://github.com/tpylman90-ctrl/Forge-Core/tree/archive/pre-open-3d-2026-11-25).

The initial milestone is one reproducible structured request producing an editable Blender model, textured GLB, previews and an independent validation report. This is not yet implemented.
