# Forge Core — ChatGPT companion 3D training plan

## Product role

Forge is a specialized, AI-driven 3D production aid that works alongside ChatGPT. ChatGPT (or another compatible assistant) handles conversation, open-ended reasoning and translating a creator's intent into a structured request. Forge handles the 3D-specific work: visual analysis, editable geometry, surfaces and materials, Blender execution, inspection, targeted revisions and deliverable export.

Forge must not spend training capacity reproducing general language-model abilities. Its useful learned capability is turning images and structured constraints into valid, editable, visually faithful assets, then improving those assets through measurable 3D inspection. The interface is a future plugin/tool boundary; an actual ChatGPT plugin or MCP connection is not yet implemented.

## Training target

Train a visual-to-geometry system that predicts a bounded, versioned asset specification or sequence of geometry operations consumed by the Blender backend. Keep the result editable. Do not optimize only for a convincing single render or an opaque mesh.

Training examples should pair authorized reference images or Blender-generated views with, where available:

- the source scene and mesh, object/part graph, and parameterized construction recipe;
- camera, scale and dimension metadata;
- foreground and part masks, depth, normals and material labels;
- fixed multi-view renders and independent inspection reports;
- optional revision pairs that identify a region, the requested change, and the measured before/after result.

Synthetic Blender data is the first controlled source because it supplies exact geometry and labels. Add independently authored or licensed object families only after provenance, redistribution rights and holdout rules are recorded. Keep complete templates and object families out of training when they are assigned to evaluation.

## Model and tool roles

1. **Visual encoder:** Start by evaluating a frozen pretrained image encoder such as DINOv2. Compare it to a small image baseline. Fine-tune only a bounded adapter or selected layers if the held-out results justify the added compute.
2. **Geometry decoder:** Predict structured parts and editable operations, not unrestricted code. Train the decoder against known geometry and rendered views. The existing v1–v5 predictor is a limited synthetic prototype; its category and parameter metrics do not certify general image-to-3D ability.
3. **Blender executor:** Convert validated operations into editable scenes, then produce GLB and fixed-angle previews. Execution remains bounded and separate from the learned model.
4. **Inspector and revision loop:** Measure dimensions, silhouette, cross-sections, topology, surface continuity, materials and export validity. Use the measurements to select targeted revisions. Preserve the original and all intermediate versions.
5. **ChatGPT/tool interface:** Accept a versioned structured asset request and return a job status, previews, editable source, GLB and machine-readable inspection report. Test this interface as software integration; do not train a Forge language model to duplicate ChatGPT.

## How to use pretrained 3D systems

Use TripoSR first as an independently run single-image reconstruction baseline, alongside Forge's own Blender and image models. Compare output geometry, surface quality, consistency, editability, runtime and memory. Its outputs may be considered as pseudo-labels only in a separately identified experiment after a rights and quality review; never treat them as ground truth or mix them into the main benchmark without provenance.

A hosted Tripo API is optional and separate from Kaggle compute. Use it only when available credits and terms are verified; it does not consume Forge's free GPU hours. Do not make a paid hosted service a required Forge dependency.

Use DINOv2 as a candidate visual representation, not as a 3D generator. Keep all teacher, backbone and adapter versions pinned and licensed. Record the source and terms for every model, output and dataset.

## GPU experiment sequence

Before training, audit and disable obsolete language/assistant training triggers, verify the Kaggle GPU runtime and durable artifact export, and make a small end-to-end dataset smoke run. The current repository says this trigger audit is not yet certified; do not launch a legacy workflow.

Then use bounded, separately checkpointed runs:

1. **Baseline and runtime check (up to 1 GPU hour):** compare a simple image baseline, frozen DINOv2 features, and TripoSR inference on a small fixed set. Record wall time, peak memory, output validity and baseline metrics.
2. **Frozen encoder study (up to 2 GPU hours):** train only the structured geometry decoder on the licensed synthetic training split. Compare against the no-image median baseline and the existing random-init result.
3. **Adapter study (up to 4 GPU hours):** only if the frozen encoder beats those baselines, compare a small adapter/partial fine-tune under the same data, compute and evaluation protocol.
4. **Repeat and refine:** spend remaining available hours only on a measured gain, independent seeds, a stronger held-out family test, or a targeted failure analysis. Keep an explicit reserve for reruns and artifacts.

These are budget ceilings for a proposed first tranche, not guaranteed runtime or a training claim. Confirm available Kaggle quota at run time. Stop early on runtime incompatibility, rights uncertainty, invalid outputs or sustained regression. Do not consume the quota just to exhaust it.

## Evaluation gates

Report image-only performance separately from any image-plus-structured-constraints condition. Hold out whole object templates and families. Compare every learned result with a no-image baseline and the prior visual checkpoint where applicable.

Track at least:

- geometry: dimension error, Chamfer distance or an equivalent mesh-distance metric, part completeness and cross-section error;
- rendered fidelity: multi-view silhouette overlap and depth/normal consistency;
- production quality: topology and non-manifold checks, surface continuity, material/texture fidelity and GLB validation;
- usefulness: whether a user can select and revise a region without rebuilding the asset; tool request success, export success, runtime and memory;
- uncertainty: seed variation, failures, and performance on genuinely unseen families.

A gain in image classification, category recognition or synthetic parameter MAE alone does not pass the asset-generation gate. No checkpoint is promoted automatically. Promotion requires repeatable held-out gains, successful Blender reconstruction and a separate review of data/model rights.

## Scope and current status

The active goal is an image-conditioned, editable 3D asset workflow that can later be called as a companion tool by ChatGPT. Blender construction, inspection and exports are the initial production path. A custom geometry engine and the assistant interface remain future work.

Existing visual experiments and the Kaggle mesh-bounds pilot remain research artifacts. This plan does not claim that Forge already generates arbitrary photo-derived, professional-quality assets. No GPU run, model promotion, plugin connection or legacy workflow deactivation is implied by this document.
