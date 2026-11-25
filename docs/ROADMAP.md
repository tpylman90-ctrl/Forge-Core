# Forge Core — Open 3D roadmap v2

All milestones below are proposed unless explicitly marked implemented. The former language-first roadmap is preserved on the [archive branch](https://github.com/tpylman90-ctrl/Forge-Core/tree/archive/pre-open-3d-2026-11-25).

| Gate | Deliverable | Acceptance evidence |
|---|---|---|
| G0 — transition | New public charter, active scope, lane isolation and archive | Versioned documents and preserved prior commit |
| G1 — inspection | GLB ingestion, original untouched renders, dimensions, cross-sections, topology and curvature | Reference fixtures; reproducible reports; original file hash unchanged |
| G2 — construction | Versioned structured asset requests and bounded Blender executor | Editable scene and valid GLB from a simple asset request; repeatable output |
| G3 — materials | UV mapping, PBR material creation and texture baking | Embedded textures, material checks and multi-view previews |
| G4 — refinement | Region-specific edits, topology repair and geometric constraints | Original and revised models plus comparison renders and metrics |
| G5 — production | Mesh budgets, LODs and Unity/Godot import checks | Independently verified imports and measured asset budgets |
| G6 — access | Local CLI, accessible phone workflow and optional assistant/MCP interface | End-to-end user test, explicit permissions, no proprietary assistant dependency |
| G7 — native engine | Backend-independent geometry operations and custom kernel | Parity benchmarks against Blender; no premature backend replacement |

## Initial benchmark assets

Start with a modular crate and progressively add a complex hard-surface vehicle, architectural prop and organic object. Distinguish procedural, licensed and independently sourced examples. Hold out whole object families and measure topology, dimensions, silhouettes, surface quality and material fidelity separately. Do not claim arbitrary image-to-3D reconstruction from primitive regressions.

## Compute

CPU: deterministic analysis and bounded baselines. GPU: image/shape and material research. Experimental: isolated representation and algorithm studies. See [lane policy](OPEN-3D-TRAINING.md).

Prioritize inspection and construction before expensive training. No old checkpoint promotion or legacy language continuation.
