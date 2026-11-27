"""Train and inspect surface reconstruction on procedural open-part geometry."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from cad_shape_v1 import GRID, Net, iou, mode_input, run, sha
from cpu_polyhaven import poly_mesh
from scaled_cad_v2 import materialize, points

LAYOUTS = {
    "train": ("table", "stool", "shelving", "open-frame", "cabinet", "pedestal"),
    "valid": ("table-lower-brace", "frame-double", "cabinet-recess"),
    "test": ("table-high-brace", "shelving-offset", "cabinet-stepped"),
}


def source(layout, index, split):
    seed = sum((i + 1) * ord(c) for i, c in enumerate(layout))
    rng = np.random.default_rng([20261127, seed, index])
    w, h, d = rng.uniform(.70, 1.25), rng.uniform(.42, .82), rng.uniform(.42, .82)
    if layout == "pedestal":
        w *= .65
    thickness = min(w, h, d) * rng.uniform(.09, .18)
    inset = thickness * rng.uniform(.52, 1.15)
    parts = []

    def box(extents, center):
        part = trimesh.creation.box(extents=extents)
        part.apply_translation(center)
        parts.append(part)

    box((w, thickness, d), (0, h - thickness / 2, 0))
    if layout.startswith("cabinet") or layout == "pedestal":
        width = w * (.42 if layout == "pedestal" else .84)
        box((width, h - thickness, d * .84), (0, (h - thickness) / 2, 0))
    else:
        for x in (-w / 2 + inset, w / 2 - inset):
            for z in (-d / 2 + inset, d / 2 - inset):
                box((thickness, h - thickness, thickness), (x, (h - thickness) / 2, z))
    if layout in ("shelving", "open-frame", "frame-double", "shelving-offset"):
        shelf_heights = (.15, .44, .72) if layout.startswith("shelving") else (.14,)
        if layout == "frame-double":
            shelf_heights = (.15, .48)
        for level in shelf_heights:
            depth = d * (.72 if layout == "shelving-offset" and level > .4 else 1.)
            box((w, thickness, depth), (0, h * level, (d - depth) / 2))
    if layout.startswith("table-"):
        level = .23 if layout == "table-lower-brace" else .57
        box((w - 2 * inset, thickness, thickness), (0, h * level, 0))
        for x in (-w / 2 + inset, w / 2 - inset):
            box((thickness, thickness, d - 2 * inset), (x, h * level, 0))
    if layout == "cabinet-recess":
        box((w * .62, thickness, d), (0, h * .12, 0))
    if layout == "cabinet-stepped":
        box((w, h * .24, d), (0, h * .12, 0))
    ident = f"{layout}:{index}"
    return {"id": ident, "family": layout, "source": "Forge procedural geometry",
            "license": "CC0-1.0", "split": split,
            "cloud": points(trimesh.util.concatenate(parts).triangles, ident)}


def empty_space_objective(logits, targets):
    # A miss in any orthographic silhouette identifies a known-empty ray.
    occupied_rays = (targets.amax(dim=1, keepdim=True).bool()
                     & targets.amax(dim=2, keepdim=True).bool()
                     & targets.amax(dim=3, keepdim=True).bool())
    weights = torch.where(occupied_rays, 1., 2.)
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, weight=weights, pos_weight=logits.new_tensor(5.)
    )
    pred = logits.sigmoid()
    axes = (1, 2, 3)
    dice = (2 * (pred * targets).sum(axes) + 1) / (pred.sum(axes) + targets.sum(axes) + 1)
    return bce + .5 * (1 - dice.mean())


def inspect_table(root):
    row = poly_mesh("WoodenTable_01", {}, root)
    index = np.clip(np.floor((row.pop("cloud") + .5) * (GRID - 2) + 1).astype(int), 0, GRID - 1)
    target = np.zeros((GRID,) * 3, dtype=bool)
    target[tuple(index.T)] = True
    views = np.stack([target.max(axis) for axis in range(3)])
    x = torch.from_numpy(views[None]).float()
    reference = torch.from_numpy(target[None])
    net = Net().eval()
    selected = next(root.glob("seed-*/selected.pt"))
    net.load_state_dict(torch.load(selected, map_location="cpu", weights_only=True)["model"])
    report = {"asset": row, "checkpoint_sha256": sha(selected), "modes": {}}
    with torch.inference_mode():
        for mode in ("single", "multi"):
            prediction = net(x, mode_input(x, mode)).sigmoid() >= .5
            positions = np.argwhere(prediction[0].numpy())
            report["modes"][mode] = {"surface_iou": iou(prediction, reference).item(),
                                     "predicted_voxels": len(positions)}
            if len(positions):
                trimesh.voxel.ops.multibox(
                    (positions + .5) / GRID - .5, pitch=1 / GRID, remove_internal_faces=True
                ).export(root / f"table-{mode}.glb")
    (root / "table.json").write_text(json.dumps(report, indent=2))
    print("TABLE " + json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.out.mkdir()
    counts = {"train": 96, "valid": 24, "test": 24}
    rows = [source(layout, index, split)
            for split, layouts in LAYOUTS.items() for layout in layouts
            for index in range(counts[split])]
    data, records, hashes = materialize(args.out, rows, 1, 1)
    manifest = args.out / "dataset_manifest.json"
    manifest.write_text(json.dumps({
        "records": records, "hashes": hashes, "grid": GRID,
        "objective": "surface BCE, known empty ray weighting, soft Dice",
    }, indent=2))
    run(data, args.out, args.seed, 24, 24, sha(manifest), device="cpu",
        objective=empty_space_objective,
        limitations="Procedural 32-cubed surfaces with held-out layouts; no arbitrary-photo evaluation.")
    inspect_table(args.out)


if __name__ == "__main__":
    main()
