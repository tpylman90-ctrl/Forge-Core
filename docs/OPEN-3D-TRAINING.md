# Active training lanes — Open 3D

| Lane | Role | Default |
|---|---|---|
| CPU | Geometry analysis, deterministic tools, small vision baselines | Bounded; test first |
| GPU | Image/depth/shape learning, reconstruction and materials | Explicit run approval and budget |
| Experimental | Alternative representations, geometry kernels and algorithms | Isolated from production |

Every run records dataset IDs, licenses, hashes, split strategy, code commit, seed, compute budget, parent and candidate checkpoint hashes, optimizer state where applicable, held-out metrics, and artifact locations. Retain periodic checkpoints and stop on measured regression against a predeclared threshold. Promote only after separate review and repeatable improvement. Record negative results as well as successes. Treat expiring artifacts as temporary, not durable archives.

Keep existing visual candidates research-only. Prior language, dialogue, personal memory, general reasoning, and assistant-oriented teacher lanes are out of scope; do not dispatch them. Their historical sources remain intact. Existing workflow definitions still need a full trigger/dependency audit before technical deactivation can be certified. No training starts merely because this document changes.

## Benchmark policy

Use licensed or independently authored data. Hold out whole object families and independently check dimensions, geometry, cross-sections, surface continuity, silhouette, topology and texture fidelity. Report simple nonlearned baselines and multi-seed variation. Keep evaluation and training splits separate. Verify dataset and checkpoint redistribution rights before public release.
