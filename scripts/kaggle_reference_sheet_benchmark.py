#!/usr/bin/env python3
"""Prepare and benchmark a private reference-sheet → image-to-3D run.

The script is designed for Kaggle. It never trains model weights and never
copies reference images into its result bundle. The default output contains
only generated GLBs, a run manifest, and geometry summary metrics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


REQUIRED_VIEWS = {"front", "top", "side"}
PREFERRED_VIEW_ORDER = ("front", "rear", "top", "side", "three_quarter")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_spec(spec: dict[str, Any]) -> None:
    if spec.get("schema_version") != 1:
        raise ValueError("reference spec schema_version must be 1")
    if not isinstance(spec.get("asset_id"), str) or not spec["asset_id"].strip():
        raise ValueError("asset_id must be a non-empty string")
    if spec.get("units") not in {"m", "cm", "mm", "in", "ft"}:
        raise ValueError("units must be one of m, cm, mm, in, ft")
    dims = spec.get("dimensions")
    if not isinstance(dims, dict):
        raise ValueError("dimensions object is required")
    for key in ("length", "width", "height"):
        value = dims.get(key)
        if value is not None and (not isinstance(value, (int, float)) or value <= 0):
            raise ValueError(f"dimensions.{key} must be positive or null")
    frame = spec.get("coordinate_frame")
    if not isinstance(frame, dict) or not all(frame.get(k) for k in ("x", "y", "z", "forward_axis")):
        raise ValueError("coordinate_frame must specify x, y, z, and forward_axis")
    views = spec.get("views")
    if not isinstance(views, dict):
        raise ValueError("views object is required")
    for label, view in views.items():
        if not isinstance(view, dict):
            raise ValueError(f"views.{label} must be an object")
        crop = view.get("crop_norm")
        if not isinstance(crop, list) or len(crop) != 4:
            raise ValueError(f"views.{label}.crop_norm must be [left, top, width, height]")
        left, top, width, height = crop
        if any(not isinstance(v, (int, float)) for v in crop):
            raise ValueError(f"views.{label}.crop_norm values must be numeric")
        if left < 0 or top < 0 or width <= 0 or height <= 0 or left + width > 1 or top + height > 1:
            raise ValueError(f"views.{label}.crop_norm must stay inside [0,1]")
        confidence = view.get("confidence")
        if confidence is not None and (not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1):
            raise ValueError(f"views.{label}.confidence must be between 0 and 1")
    missing = REQUIRED_VIEWS - set(views)
    if missing:
        raise ValueError(f"reference sheet is missing required views: {', '.join(sorted(missing))}")
    curvature = spec.get("curvature")
    if not isinstance(curvature, dict) or not isinstance(curvature.get("present"), bool):
        raise ValueError("curvature.present boolean is required")
    if curvature.get("present"):
        if not isinstance(curvature.get("legend_present"), bool):
            raise ValueError("curvature.legend_present boolean is required when curvature is present")
        if not curvature["legend_present"] and curvature.get("color_semantics") is not None:
            raise ValueError("curvature colors cannot be interpreted when no legend is visible")


def crop_views(spec: dict[str, Any], spec_path: Path, out_dir: Path) -> dict[str, Path]:
    """Crop normalized view boxes from the private source sheet."""
    validate_spec(spec)
    source_name = spec.get("source_sheet")
    if not isinstance(source_name, str) or not source_name:
        raise ValueError("source_sheet must name the private input image")
    source_path = (spec_path.parent / source_name).resolve()
    if spec_path.parent.resolve() not in source_path.parents:
        raise ValueError("source_sheet must stay inside the private reference input directory")
    if not source_path.is_file():
        raise FileNotFoundError(f"source sheet not found beside spec: {source_name}")
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Install Pillow in Kaggle before cropping views") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    with Image.open(source_path) as sheet:
        sheet = sheet.convert("RGB")
        image_w, image_h = sheet.size
        for label in PREFERRED_VIEW_ORDER:
            view = spec["views"].get(label)
            if not view:
                continue
            left, top, width, height = view["crop_norm"]
            box = (
                round(left * image_w), round(top * image_h),
                round((left + width) * image_w), round((top + height) * image_h),
            )
            output = out_dir / f"{label}.png"
            sheet.crop(box).save(output)
            outputs[label] = output
    return outputs


def sorted_aspect_error(extents: list[float], dimensions: dict[str, float | None]) -> float | None:
    """Scale-independent envelope error; it is not a shape similarity score."""
    expected = [dimensions.get(k) for k in ("length", "width", "height")]
    if any(v is None for v in expected):
        return None
    actual_sorted = sorted((float(v) for v in extents), reverse=True)
    expected_sorted = sorted((float(v) for v in expected), reverse=True)
    actual_norm = [v / actual_sorted[0] for v in actual_sorted]
    expected_norm = [v / expected_sorted[0] for v in expected_sorted]
    return sum(abs(a - b) for a, b in zip(actual_norm, expected_norm)) / 3


def inspect_glb(path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("Install trimesh in Kaggle to inspect generated GLBs") from exc
    loaded = trimesh.load(path, force="scene")
    geometries = [g for g in loaded.geometry.values() if hasattr(g, "faces") and len(g.faces)]
    if not geometries:
        raise ValueError(f"generated GLB has no triangle mesh: {path.name}")
    combined = trimesh.util.concatenate(geometries)
    extents = [float(v) for v in combined.extents]
    if any(v <= 0 for v in extents):
        raise ValueError(f"generated GLB has a zero-sized axis: {path.name}")
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "vertex_count": int(len(combined.vertices)),
        "triangle_count": int(len(combined.faces)),
        "watertight": bool(combined.is_watertight),
        "extents_sorted": sorted(extents, reverse=True),
        "aspect_ratio_mae": sorted_aspect_error(extents, spec["dimensions"]),
        "aspect_ratio_note": "sorted bounding-box ratio only; does not measure silhouette, topology quality, or design fidelity",
    }


def _save_glb(pipeline: Any, outputs: dict[str, Any], path: Path, simplify: float, texture_size: int) -> None:
    from trellis.utils import postprocessing_utils
    glb = postprocessing_utils.to_glb(
        outputs["gaussian"][0], outputs["mesh"][0],
        simplify=simplify, texture_size=texture_size, verbose=False,
    )
    glb.export(str(path))


def run_trellis(args: argparse.Namespace, spec: dict[str, Any], crops: dict[str, Path], out_dir: Path) -> list[dict[str, Any]]:
    actual_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.trellis_root,
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if actual_revision.lower() != args.source_revision.lower():
        raise ValueError(f"TRELLIS checkout is {actual_revision}, expected pinned revision {args.source_revision}")
    sys.path.insert(0, str(args.trellis_root.resolve()))
    try:
        import torch
        from PIL import Image
        from trellis.pipelines import TrellisImageTo3DPipeline
    except ImportError as exc:
        raise RuntimeError("Run this stage inside the prepared Kaggle TRELLIS environment") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle GPU is required for the TRELLIS generation stage")
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if total_vram < 15.0:
        raise RuntimeError(f"Detected {total_vram:.1f} GiB GPU memory; TRELLIS image generation expects about 16 GiB")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub in the pinned Kaggle environment") from exc
    model_path = snapshot_download(repo_id=args.model_id, revision=args.model_revision)
    pipeline = TrellisImageTo3DPipeline.from_pretrained(model_path)
    pipeline.cuda()
    run_rows = []
    multi_labels = [v for v in PREFERRED_VIEW_ORDER if v in crops]
    if len(multi_labels) < 4 or "three_quarter" not in crops:
        raise ValueError("multi-view generation requires front, top, side, and a three-quarter view")
    preferred_single = "three_quarter"
    for seed in args.seeds:
        variants = {
            "single_view": [preferred_single],
            "multi_view": multi_labels,
        }
        for variant, labels in variants.items():
            images = [Image.open(crops[label]).convert("RGB") for label in labels]
            if variant == "single_view":
                outputs = pipeline.run(
                    images[0], seed=seed, formats=["mesh", "gaussian"],
                    sparse_structure_sampler_params={"steps": args.steps, "cfg_strength": 7.5},
                    slat_sampler_params={"steps": args.steps, "cfg_strength": 3.0},
                )
            else:
                outputs = pipeline.run_multi_image(
                    images, seed=seed, formats=["mesh", "gaussian"], mode=args.multi_image_mode,
                    sparse_structure_sampler_params={"steps": args.steps, "cfg_strength": 7.5},
                    slat_sampler_params={"steps": args.steps, "cfg_strength": 3.0},
                )
            stem = f"{variant}-seed-{seed}"
            glb_path = out_dir / f"{stem}.glb"
            _save_glb(pipeline, outputs, glb_path, args.simplify, args.texture_size)
            metrics = inspect_glb(glb_path, spec)
            metrics.update({"variant": variant, "seed": seed, "views_used": labels})
            run_rows.append(metrics)
            del outputs, images
            torch.cuda.empty_cache()
    return run_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True, help="Private GPT-vision reference JSON beside the source sheet")
    parser.add_argument("--out", type=Path, required=True, help="Kaggle working output directory")
    parser.add_argument("--prepare-only", action="store_true", help="Validate and crop input without downloading or running TRELLIS")
    parser.add_argument("--model-id", default="microsoft/TRELLIS-image-large")
    parser.add_argument("--model-revision", help="Reviewed 40-character Hugging Face model commit SHA")
    parser.add_argument("--model-license", default="unreviewed", help="Record the exact model license reviewed for this private research run")
    parser.add_argument("--trellis-root", type=Path, help="Path to the pinned TRELLIS checkout")
    parser.add_argument("--source-revision", help="Reviewed 40-character TRELLIS source commit SHA")
    parser.add_argument("--seeds", type=int, nargs="+", default=[43], help="Use 43 53 73 for a three-seed comparison")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--multi-image-mode", choices=("multidiffusion", "stochastic"), default="multidiffusion")
    parser.add_argument("--simplify", type=float, default=0.95)
    parser.add_argument("--texture-size", type=int, default=1024)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    validate_spec(spec)
    if not args.prepare_only:
        if args.trellis_root is None or not args.trellis_root.is_dir():
            raise ValueError("--trellis-root must point to the pinned TRELLIS checkout")
        for name, value in (("model-revision", args.model_revision), ("source-revision", args.source_revision)):
            if not isinstance(value, str) or len(value) != 40 or any(c not in "0123456789abcdef" for c in value.lower()):
                raise ValueError(f"--{name} must be a reviewed 40-character commit SHA")
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        crops = crop_views(spec, args.spec.resolve(), out_dir / "private-crops")
        print(json.dumps({"status": "input_ready", "cropped_views": sorted(crops), "source_sha256": sha256_file((args.spec.parent / spec["source_sheet"]).resolve())}, indent=2))
        return 0

    source_path = (args.spec.parent / spec["source_sheet"]).resolve()
    with tempfile.TemporaryDirectory(prefix="forge-private-crops-") as crop_tmp:
        crops = crop_views(spec, args.spec.resolve(), Path(crop_tmp))
        metrics = run_trellis(args, spec, crops, out_dir)
    report = {
        "schema_version": 1,
        "status": "research_only_not_promoted",
        "asset_id": spec["asset_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sheet_sha256": sha256_file(source_path),
        "model": {"id": args.model_id, "revision": args.model_revision, "license_review": args.model_license},
        "trellis_source_revision": args.source_revision,
        "generation": {"seeds": args.seeds, "steps": args.steps, "multi_image_mode": args.multi_image_mode},
        "source_images_copied_to_report": False,
        "outputs": metrics,
        "interpretation": "Compare single-view and multi-view artifacts with human visual review. Bounding-box aspect error is only a scale-independent proxy, not proof of 3D fidelity.",
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
