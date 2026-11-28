# Active training lanes — Open 3D

Forge is a specialized 3D production aid designed to work alongside ChatGPT and compatible assistants. The assistant owns open-ended conversation and reasoning; Forge owns 3D visual analysis, editable geometry, Blender execution, materials, inspection, revision and asset export. Training must improve those 3D capabilities. Forge does not train a general language model or duplicate the assistant.

| Lane | Role | Default |
|---|---|---|
| CPU | Dataset validation, geometry inspection, deterministic Blender jobs, smoke tests and small baselines | Bounded; test first |
| GPU | Image/depth/shape understanding, image-conditioned geometry, reconstruction and materials | Explicit run approval and compute budget |
| Experimental | Alternative representations, pretrained visual encoders, teacher comparisons and geometry algorithms | Isolated from production |

## Active training objective

Train image-conditioned models to produce a validated, editable asset specification or bounded geometry-operation sequence for the Blender backend. Optimize for usable 3D output, not category recognition alone. Keep tool orchestration and assistant-facing interface schemas as software integration concerns; do not spend training compute on language or dialogue capabilities.

Use paired images and known Blender scenes/geometry as the first controlled source. Where available, include part graph, dimensions, camera, masks, depth, normals, materials and construction parameters. Preserve complete object-family holdouts. Add independently sourced/licensed data only after recording provenance and redistribution rights.

Evaluate frozen pretrained visual encoders (DINOv2 is a candidate) before fine-tuning. Use TripoSR as an independent single-image reconstruction baseline. It is not Forge's geometry decoder and its output is not ground truth. Any pseudo-label experiment must be isolated, rights-reviewed and compared against verified geometry. Hosted services such as the Tripo API are optional and separate from free GPU-hour use.

See [ChatGPT companion 3D training plan](CHATGPT-COMPANION-3D-TRAINING.md) for the ordered GPU experiment and evaluation gates.

## Reproducibility and checkpoints

Every run records dataset IDs, licenses, hashes, split strategy, code commit, seed, compute budget, parent and candidate checkpoint hashes, optimizer state where applicable, held-out metrics, and artifact locations. Retain periodic checkpoints and stop on measured regression against a predeclared threshold. Promote only after separate review and repeatable improvement. Record negative results as well as successes. Treat expiring artifacts as temporary, not durable archives.

Keep existing visual candidates research-only. Prior language, dialogue, personal memory, general reasoning, and assistant-oriented teacher lanes are out of scope; do not dispatch them. Their historical sources remain intact. Existing workflow definitions still need a full trigger/dependency audit before technical deactivation can be certified. No training starts merely because this document changes.

## Benchmark policy

Use licensed or independently authored data. Hold out whole object families and templates. Measure dimensions, geometry distance, part completeness, cross-sections, surface continuity, silhouette/depth consistency, topology, material fidelity, editability, export validity, runtime and memory. Report no-image baselines, prior visual checkpoint comparisons and multi-seed variation. Keep evaluation and training splits separate. Verify dataset and checkpoint redistribution rights before public release.

A category, part-map or synthetic-parameter improvement alone is not evidence of a useful image-to-editable-3D asset pipeline.
