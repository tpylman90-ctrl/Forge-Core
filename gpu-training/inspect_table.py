"""Run Forge's archived surface checkpoint on the Poly Haven Wooden Table 01."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from trimesh.voxel.ops import multibox

from cad_shape_v1 import GRID, Net, iou, mode_input, model_sha256, sha
from cpu_polyhaven import poly_mesh


archive, out = map(Path, sys.argv[1:])
out.mkdir()
candidate = next(archive.rglob("expanded/seed-43/selected.pt"))
metrics = json.loads(candidate.with_name("metrics.json").read_text())
if sha(candidate) != metrics["checkpoint_sha256"]:
    raise ValueError("Checkpoint hash mismatch")
net = Net().eval()
state = torch.load(candidate, weights_only=True, map_location="cpu")
net.load_state_dict(state["model"])
if model_sha256(net) != metrics["selected_model_sha256"]:
    raise ValueError("Weight hash mismatch")
asset = poly_mesh("WoodenTable_01", {}, out)
# Materialize the same unaugmented reference independently for each projection.
cloud = asset.pop("cloud")
index = np.clip(np.floor((cloud + .5) * (GRID - 2) + 1).astype(int), 0, GRID - 1)
surface = np.zeros((GRID,) * 3, dtype=bool)
surface[tuple(index.T)] = True
views = np.stack([surface.max(axis=axis) for axis in range(3)])
for axis, view in enumerate(views):
    Image.fromarray((~view).astype(np.uint8) * 255).resize((256, 256), Image.Resampling.NEAREST).save(out / f"input-axis-{axis}.png")
x = torch.from_numpy(views[None]).float()
target = torch.from_numpy(surface[None])
report = {"asset": asset, "checkpoint_sha256": sha(candidate), "seed": 43, "threshold": .5, "results": {}}
reference = multibox((np.argwhere(surface) + .5) / GRID - .5, pitch=1/GRID, remove_internal_faces=True)
reference.export(out / "reference.glb")
with torch.inference_mode():
    for mode in ("single", "multi"):
        predicted = net(x, mode_input(x, mode)).sigmoid() >= .5
        voxels = predicted[0].numpy()
        report["results"][mode] = {"surface_iou": iou(predicted, target).item(), "voxel_count": int(voxels.sum())}
        if voxels.any():
            mesh = multibox((np.argwhere(voxels) + .5) / GRID - .5, pitch=1/GRID, remove_internal_faces=True)
            mesh.export(out / f"forge-{mode}.glb")
(out / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
