# Visual reference-sheet → editable 3D benchmark

## Purpose

The Forge visual experiments through v5 learned from small procedural Blender
examples. They did not demonstrate that Forge can read a real multi-view
reference sheet or reconstruct a production vehicle. This lane measures a
different workflow: GPT vision reads the sheet at runtime, then a GPU image-to-3D
tool produces a mesh that can be inspected and edited in Blender/ModelForge.
It does not train or update Forge language weights.

## Data flow

1. Keep the original reference sheet and its JSON spec in private Kaggle inputs.
   The JSON includes each view's normalized crop box, dimensions/units, axis
   convention, visible parts, confidence, and uncertainty notes.
2. Use the prompt in `prompts/REFERENCE-SHEET-VISION-V1.md` with GPT vision to
   create that structured reading. The prompt requires it to abstain on
   curvature colors when the legend is absent and to mark hidden geometry as
   unknown.
3. Run `scripts/kaggle_reference_sheet_benchmark.py` in a Kaggle GPU notebook.
   It crops the linked views, generates a single-view baseline and a multi-view
   candidate with Microsoft's TRELLIS image pipeline, exports GLBs, and records
   source/model revisions, hashes, mesh statistics and scale-independent
   bounding-box aspect error.
4. Inspect the GLBs and turntable previews in Blender/ModelForge. Record the
   human review separately; bounding-box ratios and watertightness do not measure
   whether the craft matches the reference.

The official TRELLIS repository documents multi-image generation as
experimental and warns that it may not work well for inconsistent views. The
benchmark therefore compares single-view and multi-view outputs with identical
seeds and settings. The upstream image model is documented as needing roughly
16 GB of GPU memory; the script checks available VRAM before downloading weights.

## Kaggle setup

Use a private Kaggle notebook/dataset for the supplied image. Do not put the
image, crops, API keys, or generated GLB in this public repository. Upload the
vision JSON beside the source sheet and set `source_sheet` to its filename.

Install the pinned TRELLIS environment using the upstream Linux setup
instructions and check out the same exact source commit passed as
`--source-revision`, then run. The model snapshot is resolved at the exact
`--model-revision` before TRELLIS loads it. Review and record the model's
license terms before relying on generated assets; this research lane does not
grant redistribution rights.

```bash
python scripts/kaggle_reference_sheet_benchmark.py \
  --spec /kaggle/input/private-reference/reference.json \
  --out /kaggle/working/forge-reference-run \
  --trellis-root /kaggle/working/TRELLIS \
  --model-revision <reviewed-hugging-face-commit> \
  --source-revision <reviewed-trellis-commit> \
  --model-license <reviewed-license-name> \
  --seeds 43 53 73
```

For a quick smoke run, use `--prepare-only` first. It validates the image-sheet
spec and writes view crops locally without loading the 3D model. Kaggle outputs
remain in the private working session unless the user separately chooses to
download them.

## Gates and interpretation

- Reference crop boxes must be valid; front, top, side, and one three-quarter
  view are required for the multi-view arm.
- Dimensions must carry units and a coordinate convention. Missing measurements
  remain null.
- Curvature semantics must remain null when the source image has no visible
  legend.
- The benchmark reports the single-view and multi-view runs separately, with
  identical seeds. It never treats them as a trained checkpoint or promotes a
  result into Forge.
- Multi-view improvement is a hypothesis to measure, not an expected outcome.
  A mesh that passes metadata checks can still have poor topology or incorrect
  hidden surfaces.

## GitHub checks

The `Visual reference sheet contract` Actions workflow checks the JSON/crop
contract using synthetic test data only. It does not require Kaggle credentials,
download user images, fetch large model weights, or spend GPU time. The Kaggle
GPU run remains a separate, deliberate notebook session.
