"""Compare CAD-only and expanded mesh training on the same Poly Haven objects."""
import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
import torch
import trimesh

from cad_shape_v1 import SOURCE_BLOB, SOURCE_COMMIT, SPLITS, generator, run
from scaled_cad_v2 import digest, materialize, points

API = "https://api.polyhaven.com"
LICENSE = "https://polyhaven.com/license"
LIMIT = 48 * 1024 * 1024


def fetch(url, limit=LIMIT):
    parsed = urlparse(url)
    host = parsed.hostname or ""
    if parsed.scheme != "https" or host not in {"api.polyhaven.com", "dl.polyhaven.org", "cdn.polyhaven.com"}:
        raise ValueError(f"Unexpected Poly Haven download host: {host}")
    request = Request(url, headers={"User-Agent": "Forge-Core-CPU-Shape-Research"})
    with urlopen(request, timeout=30) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Download exceeds size limit")
    return data


def poly_mesh(asset_id, metadata, work):
    files = json.loads(fetch(f"{API}/files/{asset_id}", 2**20))
    info = files["gltf"]["1k"]["gltf"]
    raw = fetch(info["url"])
    document = json.loads(raw)
    mesh_hash = hashlib.sha256(raw)
    # Load geometry only. The model takes silhouettes and surface voxels.
    for key in ("images", "textures", "samplers", "materials"):
        document.pop(key, None)
    for mesh in document["meshes"]:
        for primitive in mesh["primitives"]:
            primitive.pop("material", None)
    with tempfile.TemporaryDirectory(dir=work) as directory:
        folder = Path(directory)
        for buffer in document["buffers"]:
            name = buffer["uri"]
            if Path(name).name != name or not name.endswith(".bin"):
                raise ValueError("Unexpected glTF buffer path")
            payload = fetch(info["include"][name]["url"])
            mesh_hash.update(payload)
            (folder / name).write_bytes(payload)
        source = folder / "asset.gltf"
        source.write_text(json.dumps(document))
        mesh = trimesh.load_scene(source, process=False).to_geometry()
        if len(mesh.faces) > 500_000:
            raise ValueError("Mesh exceeds CPU face budget")
        cloud = points(mesh.triangles, f"polyhaven:{asset_id}")
    return {
        "id": f"polyhaven:{asset_id}",
        "source": "Poly Haven",
        "source_url": f"https://polyhaven.com/a/{asset_id}",
        "asset_file_url": info["url"],
        "license": "CC0-1.0",
        "license_url": LICENSE,
        "authors": metadata.get("authors", {}),
        "geometry_inputs_sha256": mesh_hash.hexdigest(),
        "cloud": cloud,
    }


def external_sources(work):
    catalog = json.loads(fetch(f"{API}/assets?t=models", 8 * 2**20))
    ordered = sorted(catalog, key=lambda name: hashlib.sha256(f"forge-poly-v1:{name}".encode()).digest())
    families, hashes, rows = set(), set(), []
    for asset_id in ordered[:60]:
        if not re.fullmatch(r"[a-z0-9_]+", asset_id):
            continue
        family = re.sub(r"(?:_[0-9]+)+$", "", asset_id)
        if family in families:
            continue
        try:
            row = poly_mesh(asset_id, catalog[asset_id], work)
        except HTTPError as exc:
            if exc.code != 404:
                raise
            print(f"Skipped {asset_id}: missing file", flush=True)
            continue
        except (KeyError, ValueError, TypeError) as exc:
            print(f"Skipped {asset_id}: {type(exc).__name__}: {exc}", flush=True)
            continue
        if row["geometry_inputs_sha256"] in hashes:
            continue
        families.add(family)
        hashes.add(row["geometry_inputs_sha256"])
        row["family"] = family
        row["split"] = "train" if len(rows) < 12 else "valid" if len(rows) < 16 else "test"
        rows.append(row)
        print(f"Accepted Poly Haven {asset_id} for {row['split']}", flush=True)
        if len(rows) == 20:
            return rows
    raise RuntimeError(f"Expected 20 Poly Haven meshes, accepted {len(rows)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.out.mkdir(parents=True)
    (args.out / "source-license.json").write_text(json.dumps({
        "external_provider": "Poly Haven",
        "license": "CC0-1.0",
        "license_url": LICENSE,
        "api_terms": "https://github.com/Poly-Haven/Public-API/blob/master/ToS.md",
    }, indent=2))
    mod = generator(args.generator)
    cad = []
    for ident in SPLITS["train"]:
        triangles = np.asarray(mod.make_asset(ident).data, dtype=np.float32)
        triangles = triangles.reshape(-1, 6)[:, :3].reshape(-1, 3, 3)
        cad.append({
            "id": f"cad:{ident}", "source": "Forge Basics", "split": "train",
            "license": "CC0-1.0", "revision": SOURCE_COMMIT, "generator_blob": SOURCE_BLOB,
            "cloud": points(triangles, ident),
        })
    external = external_sources(args.out)
    holdout = [row for row in external if row["split"] != "train"]
    datasets = {}
    for arm, rows, cad_variants in (
        ("cad-only", cad + holdout, 16),
        ("expanded", cad + external, 8),
    ):
        root = args.out / arm
        root.mkdir()
        data, records, hashes = materialize(root, rows, cad_variants, 8)
        manifest = root / "dataset_manifest.json"
        manifest.write_text(json.dumps({
            "records": records, "hashes": hashes,
            "source_object_counts": {split: len(items) for split, items in records.items()},
            "sample_counts": {split: hashes[split]["samples"] for split in records},
        }, indent=2))
        datasets[arm] = (data, root, digest(manifest), hashes)
    a, b = (datasets[arm][3] for arm in ("cad-only", "expanded"))
    if a["valid"] != b["valid"] or a["test"] != b["test"] or a["train"]["samples"] != b["train"]["samples"]:
        raise RuntimeError("Comparison datasets do not match the protocol")
    reports = {}
    limits = "Small source-disjoint Poly Haven sample; coarse surface voxels; near duplicates may remain."
    for arm, (data, root, manifest_hash, _) in datasets.items():
        reports[arm] = [
            run(data, root, seed, 8, 16, manifest_hash, device="cpu", limitations=limits)
            for seed in (43, 53, 73)
        ]
        for path in root.glob("*.npy"):
            path.unlink()
    (args.out / "comparison.json").write_text(json.dumps(reports, indent=2))
    print("Held-out surface IoU, CAD-only to expanded (higher is better)", flush=True)
    for baseline, expanded in zip(reports["cad-only"], reports["expanded"]):
        print(expanded["seed"], {
            mode: [baseline["test"][mode]["surface_iou"], expanded["test"][mode]["surface_iou"]]
            for mode in ("single", "multi")
        }, flush=True)


if __name__ == "__main__":
    main()
