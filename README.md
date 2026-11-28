> **If we genuinely want technology to benefit humanity, we shouldn't just democratize access to finished technology. We should democratize the ability to create it.**

# Forge Core — Open 3D Intelligence

## Our mission

To democratize professional-quality 3D creation by developing free, open and accessible intelligent modeling tools that empower people to transform their ideas into reality, regardless of their education, technical experience or financial resources.

The next great idea should not be limited by access to schooling, expensive software or professional training. Forge aims to help creators turn ideas into editable, fully textured 3D assets.

## What we are building

Forge is an independent, AI-driven 3D production aid designed to work alongside ChatGPT and compatible assistants. The assistant handles open-ended conversation and helps shape the creator's intent into a structured request; Forge handles 3D-specific visual analysis, editable geometry, materials, Blender execution, inspection, revision and export. Forge is not being trained as a general-purpose conversational model. Blender is the initial modeling backend; a purpose-built, AI-native geometry engine is a long-term objective.

Creator ↔ ChatGPT or compatible assistant → structured asset request → Forge geometry, surfaces and materials → Blender execution and inspection → editable asset, GLB, previews and report

| Component | Purpose |
|---|---|
| Vision | Reference-image, depth, proportion and silhouette analysis |
| Geometry | Structured shape construction and precise regional edits |
| Surface | Curvature, cross-sections, topology and panel transitions |
| Materials | UVs, PBR materials, texture maps and baking |
| Inspector | Reproducible geometry and visual quality checks |
| Optimizer | LODs, mesh budgets and game-engine compatibility |
| Integration | Versioned asset requests; future plugin/MCP-compatible interface |
| Native engine | Future headless, AI-oriented modeling backend |

These are target capabilities, not claims of completed functionality. ChatGPT/plugin/MCP integration is planned and does not yet exist.

## Current status

Existing research includes procedural Blender object generation, visual neural experiments v1–v5, part maps and silhouette evaluation. A separate Kaggle pilot explored mesh bounds from images. Results remain research-only; neither full image-to-3D nor professional textured asset generation has been demonstrated.

- [Roadmap](docs/ROADMAP.md) · [Architecture](docs/ARCHITECTURE.md) · [Master blueprint](docs/MASTER-BLUEPRINT.md)
- [ChatGPT companion 3D training plan](docs/CHATGPT-COMPANION-3D-TRAINING.md) · [Training lanes and reproducibility](docs/OPEN-3D-TRAINING.md) · [Project scope](config/open_3d_scope.json)
- [Previous visual results](docs/OBJECT-PHOTO-TEXT-V4-RESULTS-2026-11-17.md) · [November 20 results](docs/FORGE-TRAINING-RESULTS-2026-11-20.md)
- [Founding charter](CHARTER.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

## Dedicated Blender workspace

[Workspace source and instructions](blender-workspace/README.md) · [Run Blender through GitHub Actions](https://github.com/tpylman90-ctrl/Forge-Core/actions/workflows/blender-workspace.yml). A separate headless, CPU-based job runner executes registered asset requests, validates GLB exports and packages editable Blender scenes, preview renders and hashes. No training weights, paid services or credentials are required. First execution verification is pending.

## Development priorities

1. Geometry understanding and reproducible shape benchmarks.
2. A bounded, structured Blender asset construction pipeline.
3. Automated inspection, targeted revisions and texture baking.
4. Game-ready GLB exports with preview images and validation reports.
5. Accessible mobile workflows and a future assistant/plugin interface.
6. Eventually, an independent geometry backend.

CPU, GPU and experimental tracks are isolated. No automatic checkpoint promotion; retain lineage and report failures.

## Open collaboration

We welcome contributions toward accessible creation, reproducibility and interoperability. Prefer free tools and local execution where practical, with a mobile-friendly path for people without a workstation. Compute requirements and limitations must be stated plainly.

Forge is independent and is not an official OpenAI product or partnership. ChatGPT integration is an objective, not an existing or guaranteed integration.

## Historical research

The former personal-assistant and language research mission is superseded. Its source, documentation and commit history are preserved on the [pre-transition archive branch](https://github.com/tpylman90-ctrl/Forge-Core/tree/archive/pre-open-3d-2026-11-25) and remain in the repository pending a dependency-aware migration. Do not start or continue legacy language/assistant training under the new charter. Existing workflow files are not yet technically disabled.

## License

Public source visibility does not by itself grant redistribution rights. The goal is an open-source release; a code license and third-party rights audit are pending. Publish datasets, model weights and asset bundles only when their terms permit it. Never upload credentials, private references or personal data.
