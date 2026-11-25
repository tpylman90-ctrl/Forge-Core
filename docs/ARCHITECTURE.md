# Forge Core — Open 3D architecture v2

Target design; only the existing visual research and procedural Blender components are implemented historical prototypes. The former architecture is preserved in the [archive](https://github.com/tpylman90-ctrl/Forge-Core/tree/archive/pre-open-3d-2026-11-25).

## Backend-independent flow

1. A creator supplies a specification and optional authorized references.
2. A client validates a versioned structured asset request and sets budgets.
3. Forge analyzes shapes, dimensions, semantic parts and constraints.
4. A bounded geometry executor constructs an editable scene; initially Blender Python in a restricted worker.
5. Surface and material stages handle topology, UVs, PBR textures and baking.
6. Independent inspection measures geometry and material properties and renders fixed views.
7. Requested revisions are applied to separate versions; preserve the original.
8. Export editable source, GLB, preview renders, and a machine-readable report.

## Interfaces and ownership

A future native kernel implements the same scene/operation contract as Blender. The initial boundary is a structured request, not arbitrary code supplied by an assistant. Validate input paths, size and resource limits. Do not execute untrusted Python, shell commands or imported scripts. No external network, secrets, or arbitrary filesystem access in a modeling worker by default. User approval governs publication and external actions.

A future MCP adapter may submit jobs and retrieve artifacts; no existing integration or official partnership is claimed. Direct local use must remain supported.

## Existing source mapping

- `src/forge_core/visual/object_branch_v1.py` through `object_branch_v5.py`: isolated visual research.
- `scripts/object_geometry_v1.py`: prior procedural geometry.
- `scripts/generate_object_dataset_v*.py`: prior synthetic data generation.
- `scripts/render_object_eval_v*.py` and `scripts/score_object_render_v*.py`: prior silhouette evaluation.
- `src/forge_core/neural/`: historical language research, not the new production dependency.
- `config/open_3d_scope.json`: new active scope manifest.

Do not move legacy modules until imports, tests and workflow dependencies are inventoried. New modules should live under a separate production namespace with no dependency on language checkpoints.
