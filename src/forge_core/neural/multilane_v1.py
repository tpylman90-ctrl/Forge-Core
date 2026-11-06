"""Isolated Forge Seed training lanes for language, Blender, and Godot.

Every lane begins from the same verified Forge checkpoint. The language lane
resumes its own optimizer/RNG on the pinned language corpus. Blender and Godot
get independent full-weight research branches with fresh optimizers and
version-pinned, attribution-preserving documentation corpora. Nothing merges
back to the common checkpoint automatically.
"""
from __future__ import annotations

import argparse
import copy
from html.parser import HTMLParser
import hashlib
import json
import os
from pathlib import Path
import random
import time
import urllib.request

import torch
from tokenizers import Tokenizer

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.seed_v07 import heldout
from forge_core.neural.train import make_batch

BASE_SHA256 = "a7bcdb996e440f44d7496b887b54d19324927b4296fe9ed2f0f0293c88c63a0a"
BASE_STEP = 95000
BASE_TOKENIZER_SHA256 = "4ecd7803d4c369e80bda2e74c30cfb769b7e3a942f931df83383eec9f794f04c"
MAX_SECONDS = 3600
SEQ_LEN = 128
BATCH = 4
EVAL_WINDOWS = 512
PATIENCE = 8

SOURCES = {
    "blender": {
        "version": "Blender 5.4",
        "license": "CC BY-SA 4.0 (manual prose); code samples excluded from this lane",
        "license_url": "https://docs.blender.org/manual/en/5.4/copyright.html",
        "attribution": "Blender Documentation Team, Blender 5.4 Manual",
        "urls": [
            "https://docs.blender.org/manual/en/5.4/modeling/meshes/primitives.html",
            "https://docs.blender.org/manual/en/5.4/scene_layout/object/origin.html",
            "https://docs.blender.org/manual/en/5.4/modeling/geometry_nodes/index.html",
            "https://docs.blender.org/manual/en/5.4/addons/scene_gltf2.html",
            "https://docs.blender.org/manual/en/5.4/scene_layout/object/properties/transforms.html",
            "https://docs.blender.org/manual/en/5.4/modeling/geometry_nodes/input/index.html",
        ],
    },
    "godot": {
        "version": "Godot 4.7",
        "license": "CC BY 3.0",
        "license_url": "https://docs.godotengine.org/en/4.7/about/faq.html",
        "attribution": "Juan Linietsky, Ariel Manzur, and the Godot Engine community",
        "urls": [
            "https://docs.godotengine.org/en/4.7/tutorials/scripting/gdscript/gdscript_basics.html",
            "https://docs.godotengine.org/en/4.7/tutorials/scripting/nodes_and_scene_instances.html",
            "https://docs.godotengine.org/en/4.7/tutorials/3d/procedural_geometry/arraymesh.html",
            "https://docs.godotengine.org/en/4.7/tutorials/3d/using_transforms.html",
            "https://docs.godotengine.org/en/4.7/tutorials/best_practices/scene_organization.html",
            "https://docs.godotengine.org/en/4.7/tutorials/physics/physics_introduction.html",
        ],
    },
}

# Original Forge-Core lesson cards. These are authored for this project (CC0-1.0);
# they are separate from the licensed manual prose and carry explicit why steps.
LESSONS = {
    "blender": [
        {
            "task": "Create a two-unit cube and add a clean three-segment bevel.",
            "answer": "import bpy\nbpy.ops.mesh.primitive_cube_add(size=2)\nobj = bpy.context.object\nobj.name = 'Beveled_Block'\nbpy.ops.object.transform_apply(location=False, rotation=False, scale=True)\nbevel = obj.modifiers.new('Edge_Bevel', 'BEVEL')\nbevel.width = 0.08\nbevel.segments = 3\nobj.modifiers.new('Weighted_Normals', 'WEIGHTED_NORMAL')",
            "why": "Apply non-uniform scale before setting a bevel width so the bevel remains consistent across faces. A weighted normal modifier improves hard-surface shading without changing the silhouette."
        },
        {
            "task": "Move an object's origin to the center of its geometry before export.",
            "answer": "import bpy\nobj = bpy.context.active_object\nif obj is None:\n    raise RuntimeError('Select an object first')\nbpy.context.view_layer.objects.active = obj\nobj.select_set(True)\nbpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')",
            "why": "A useful origin makes rotation, placement, and engine-side transforms predictable. The operation needs the intended object selected and active."
        },
        {
            "task": "Export only the selected object as a binary glTF file.",
            "answer": "import bpy\nobj = bpy.context.active_object\nif obj is None:\n    raise RuntimeError('Select an object first')\nbpy.ops.object.select_all(action='DESELECT')\nobj.select_set(True)\nbpy.context.view_layer.objects.active = obj\nbpy.ops.export_scene.gltf(filepath='asset.glb', export_format='GLB', use_selection=True)",
            "why": "A selection-only export prevents unrelated scene objects from entering the asset package. Check materials and scale again after importing the GLB into the target engine."
        }
    ],
    "godot": [
        {
            "task": "Load a PackedScene, instantiate it, and add the instance to the current scene.",
            "answer": "var actor_scene: PackedScene = load('res://actors/actor.tscn')\nif actor_scene == null:\n    push_error('Actor scene could not be loaded')\n    return\nvar actor := actor_scene.instantiate()\nadd_child(actor)",
            "why": "A PackedScene is a reusable resource; instantiate it to create a live node tree. Add the new root to the intended parent so it enters the active scene."
        },
        {
            "task": "Connect a Button press signal to a method in the current script.",
            "answer": "func _ready() -> void:\n    $Button.pressed.connect(_on_button_pressed)\n\nfunc _on_button_pressed() -> void:\n    print('Button pressed')",
            "why": "Signal connections let a node announce an event without owning the receiver's behavior. Connect once during setup and keep the response in a named callback."
        },
        {
            "task": "Create a triangle using ArrayMesh from three vertices.",
            "answer": "var arrays := []\narrays.resize(Mesh.ARRAY_MAX)\narrays[Mesh.ARRAY_VERTEX] = PackedVector3Array([Vector3.ZERO, Vector3.RIGHT, Vector3.UP])\narrays[Mesh.ARRAY_INDEX] = PackedInt32Array([0, 1, 2])\nvar mesh := ArrayMesh.new()\nmesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)",
            "why": "ArrayMesh stores vertex attributes in indexed array slots. The vertex and index arrays describe one triangle; normals and UVs can be added when lighting or texturing requires them."
        }
    ]
}

TASK_PROMPTS = {
    "blender": [
        "Write a Blender Python script that creates a tapered low-poly rock and gives it smooth shading.",
        "Explain how to prepare a Blender object for a clean GLB export to a game engine.",
        "Write a Blender Python script that adds a named bevel modifier to the active mesh and checks for no active object."
    ],
    "godot": [
        "Write GDScript that moves a CharacterBody3D from input in _physics_process and calls move_and_slide.",
        "Explain when to use a signal instead of a direct node path dependency in a reusable Godot scene.",
        "Write GDScript that loads a PackedScene, handles a missing resource, instantiates it, and attaches it to a parent."
    ],
}



def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


class ManualText(HTMLParser):
    """Extract readable prose while excluding code and page chrome."""

    DROP = {"script", "style", "noscript", "nav", "header", "footer", "aside",
            "pre", "code", "svg", "button"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.DROP:
            self.hidden += 1
        elif not self.hidden and tag in {"p", "li", "h1", "h2", "h3", "dt", "dd", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.DROP:
            self.hidden = max(0, self.hidden - 1)
        elif not self.hidden and tag in {"p", "li", "h1", "h2", "h3", "dt", "dd"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)

    def text(self) -> str:
        return " ".join(" ".join(self.parts).split())


def fetch_manual(lane: str, out: Path) -> dict:
    config = SOURCES[lane]
    pages, texts = [], []
    for url in config["urls"]:
        request = urllib.request.Request(url, headers={"User-Agent": "Forge-Core-Research/1.0"})
        with urllib.request.urlopen(request, timeout=45) as response:
            raw, final_url = response.read(), response.geturl()
        parser = ManualText()
        parser.feed(raw.decode("utf-8", errors="replace"))
        text = parser.text()
        if len(text) < 1500:
            raise ValueError(f"Manual page yielded too little prose: {url}")
        pages.append({"url": url, "resolved_url": final_url,
                      "html_sha256": sha256_bytes(raw), "extracted_characters": len(text)})
        texts.append(text)
    lesson_text = "\\n\\n".join(
        f"Task: {item['task']}\\nAnswer:\\n{item['answer']}\\nWhy:\\n{item['why']}"
        for item in LESSONS[lane]
    )
    lesson_bytes = lesson_text.encode("utf-8")
    train_text = "\\n\\n".join(texts[:4]) + "\\n\\n" + lesson_text
    valid_text, test_text = texts[4], texts[5]
    if len(train_text) < 8000 or min(len(valid_text), len(test_text)) < 1500:
        raise ValueError("Documentation train/validation/test split is too small")
    out.mkdir(parents=True, exist_ok=True)
    for name, text in (("train.txt", train_text), ("valid.txt", valid_text), ("test.txt", test_text)):
        (out / name).write_text(text + "\n", encoding="utf-8")
    manifest = {
        "schema": 1, "lane": lane, "version": config["version"],
        "license": config["license"], "license_url": config["license_url"],
        "attribution": config["attribution"],
        "split_policy": "source pages 1-4 train; page 5 validation; page 6 final test; whole-page split",
        "code_samples": "HTML pre/code blocks excluded; original lesson code is CC0-1.0",
        "curriculum_license": "CC0-1.0, original Forge-Core lesson cards",
        "lesson_sha256": sha256_bytes(lesson_bytes),
        "lesson_count": len(LESSONS[lane]),
        "pages": pages,
        "train_sha256": sha256_file(out / "train.txt"),
        "valid_sha256": sha256_file(out / "valid.txt"),
        "test_sha256": sha256_file(out / "test.txt"),
        "train_utf8_bytes": (out / "train.txt").stat().st_size,
        "valid_utf8_bytes": (out / "valid.txt").stat().st_size,
        "test_utf8_bytes": (out / "test.txt").stat().st_size,
    }
    (out / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "ATTRIBUTION.txt").write_text(
        f"{config['attribution']}\n{config['version']}\n"
        f"License: {config['license']}\n{config['license_url']}\n"
        "The extracted source text is provided only with this attribution and license notice.\n",
        encoding="utf-8",
    )
    return manifest


def load_language_data(source: Path) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    manifest = json.loads((source / "source-manifest.json").read_text())
    ckpt = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    for split in ("train", "valid"):
        actual = sha256_file(source / f"{split}.txt")
        assert actual == manifest[f"{split}_sha256"] == ckpt[f"{split}_sha256"]
    tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    train = torch.tensor(tokenizer.encode((source / "train.txt").read_text()).ids, dtype=torch.long)
    valid = torch.tensor(tokenizer.encode((source / "valid.txt").read_text()).ids, dtype=torch.long)
    return sha256_file(source / "tokenizer.json"), train, valid, valid, manifest


def load_manual_data(lane: str, folder: Path, tokenizer: Tokenizer):
    manifest = fetch_manual(lane, folder)
    train = torch.tensor(tokenizer.encode((folder / "train.txt").read_text()).ids, dtype=torch.long)
    valid = torch.tensor(tokenizer.encode((folder / "valid.txt").read_text()).ids, dtype=torch.long)
    test = torch.tensor(tokenizer.encode((folder / "test.txt").read_text()).ids, dtype=torch.long)
    if min(train.numel(), valid.numel(), test.numel()) <= SEQ_LEN + 1:
        raise ValueError("Manual split does not support fixed-window evaluation")
    return train, valid, test, manifest


def save_checkpoint(out: Path, model, optimizer_state, lane, parent, updates, seed,
                    data, best_loss, torch_rng_state, sample_rng_state):
    ckpt = {
        "schema": 3, "lane": lane, "parent_run": 42143390701,
        "parent_step": BASE_STEP, "parent_sha256": BASE_SHA256,
        "step": BASE_STEP + updates if lane == "language" else BASE_STEP,
        "lane_updates": updates, "seed": seed, "configuration": model.cfg.__dict__,
        "revision": parent["revision"], "tokenizer_sha256": BASE_TOKENIZER_SHA256,
        "train_sha256": data["manifest"]["train_sha256"],
        "valid_sha256": data["manifest"]["valid_sha256"],
        "model": model.state_dict(), "optimizer": optimizer_state,
        "torch_rng_state": torch_rng_state, "sample_rng_state": sample_rng_state,
        "source_manifest": data["manifest"], "best_valid_loss": best_loss,
    }
    tmp = out / "checkpoint.pt.tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, out / "checkpoint.pt")


def run_lane(lane: str, source: Path, out: Path, seconds: int):
    if lane not in {"language", "blender", "godot"}:
        raise ValueError("Unknown lane")
    if not 60 <= seconds <= MAX_SECONDS:
        raise ValueError("Each isolated lane must run from 60 to 3,600 seconds")
    parent_path = source / "checkpoint.pt"
    parent_hash = sha256_file(parent_path)
    if parent_hash != BASE_SHA256:
        raise ValueError(f"Wrong Forge parent checkpoint: {parent_hash}")
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    if parent.get("step") != BASE_STEP or parent.get("seed") != 43:
        raise ValueError("Expected the exact verified 95k Forge Seed checkpoint")
    if sha256_file(source / "tokenizer.json") != BASE_TOKENIZER_SHA256:
        raise ValueError("Base tokenizer hash mismatch")
    tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    if lane == "language":
        _, train_ids, valid_ids, test_ids, manifest = load_language_data(source)
    else:
        train_ids, valid_ids, test_ids, manifest = load_manual_data(lane, out / "source", tokenizer)

    seed = 43 if lane == "language" else (731 if lane == "blender" else 947)
    eval_windows = 2048 if lane == "language" else EVAL_WINDOWS
    torch.set_num_threads(min(2, torch.get_num_threads()))
    torch.manual_seed(seed)
    random.seed(seed)
    model = ByteTransformer(ModelConfig(**parent["configuration"]))
    model.load_state_dict(parent["model"])
    learning_rate = 3e-4 if lane == "language" else 5e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if lane == "language":
        optimizer.load_state_dict(parent["optimizer"])
        torch.set_rng_state(parent["torch_rng_state"])
        rng = torch.Generator()
        rng.set_state(parent["sample_rng_state"])
    else:
        rng = torch.Generator().manual_seed(seed + 1)

    out.mkdir(parents=True, exist_ok=True)
    model.train()
    initial_valid = heldout(model, valid_ids, windows=eval_windows, seq_len=SEQ_LEN)
    initial_test = heldout(model, test_ids, windows=eval_windows, seq_len=SEQ_LEN)
    best_valid = initial_valid
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_optimizer = copy.deepcopy(optimizer.state_dict())
    best_torch_rng = torch.get_rng_state().clone()
    best_sample_rng = rng.get_state().clone()
    started = time.monotonic()
    updates = 0
    curve = []
    while time.monotonic() - started < seconds:
        x, y = make_batch(train_ids, BATCH, SEQ_LEN, rng)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        updates += 1
        if updates % 1000 == 0:
            score = heldout(model, valid_ids, windows=eval_windows, seq_len=SEQ_LEN)
            curve.append({"updates": updates, "valid_token_loss": round(score, 8)})
            if lane == "language" or score < best_valid:
                best_valid = score
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_optimizer = copy.deepcopy(optimizer.state_dict())
                best_torch_rng = torch.get_rng_state().clone()
                best_sample_rng = rng.get_state().clone()
            current_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            selected_optimizer = optimizer.state_dict() if lane == "language" else best_optimizer
            if lane != "language":
                model.load_state_dict(best_state)
            save_checkpoint(out, model, selected_optimizer, lane, parent, updates, seed,
                            {"rng": rng, "manifest": manifest}, best_valid,
                            best_torch_rng, best_sample_rng)
            if lane != "language":
                model.load_state_dict(current_state)
            model.train()
            print(json.dumps({"lane": lane, "updates": updates,
                              "valid_token_loss": round(score, 8),
                              "selected_valid_loss": round(best_valid, 8),
                              "elapsed_seconds": round(time.monotonic() - started, 1)}),
                  flush=True)
    if lane == "language":
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_optimizer = copy.deepcopy(optimizer.state_dict())
        best_torch_rng = torch.get_rng_state().clone()
        best_sample_rng = rng.get_state().clone()
    model.load_state_dict(best_state)
    selected_valid = heldout(model, valid_ids, windows=eval_windows, seq_len=SEQ_LEN)
    final_test = heldout(model, test_ids, windows=eval_windows, seq_len=SEQ_LEN)
    final_optimizer = optimizer.state_dict() if lane == "language" else best_optimizer
    save_checkpoint(out, model, final_optimizer, lane, parent, updates, seed,
                    {"rng": rng, "manifest": manifest}, best_valid,
                    best_torch_rng, best_sample_rng)
    report = {
        "schema": 1, "status": "research_only_not_promoted", "lane": lane,
        "parent_run": 42143390701, "parent_checkpoint_sha256": parent_hash,
        "parent_step": BASE_STEP, "candidate_checkpoint_sha256": sha256_file(out / "checkpoint.pt"),
        "candidate_language_step": BASE_STEP + updates if lane == "language" else None,
        "updates": updates, "elapsed_seconds": round(time.monotonic() - started, 2),
        "parameters": model.num_parameters, "tokenizer_sha256": BASE_TOKENIZER_SHA256,
        "train_sha256": manifest["train_sha256"], "valid_sha256": manifest["valid_sha256"],
        "test_sha256": manifest.get("test_sha256", manifest["valid_sha256"]),
        "validation_windows": eval_windows,
        "validation_loss_parent": round(initial_valid, 8),
        "validation_loss_selected": round(selected_valid, 8),
        "test_loss_parent": round(initial_test, 8), "test_loss_candidate": round(final_test, 8),
        "curve": curve,
        "capability_claim": "None. Token prediction on documentation is not proof of tool competence.",
        "android_integrated": False,
        "optimizer_policy": "resumed only for language; fresh and lane-local for specialist branches",
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "lineage.json").write_text(json.dumps({
        "lane": lane, "parent_run": 42143390701, "parent_step": BASE_STEP,
        "parent_sha256": parent_hash, "candidate_sha256": report["candidate_checkpoint_sha256"],
        "lane_updates": updates, "base_shared_weights_modified": False,
        "lane_weights_isolated": True, "android_integrated": False,
    }, indent=2) + "\n")
    (out / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "tokenizer.json").write_bytes((source / "tokenizer.json").read_bytes())
    if lane == "language":
        from forge_core.neural.preschool_eval import evaluate_checkpoint
        evaluate_checkpoint(out, out / "capabilities.json")
        for filename in ("train.txt", "valid.txt"):
            (out / filename).write_bytes((source / filename).read_bytes())
        (out / "test.txt").write_bytes((source / "valid.txt").read_bytes())
        (out / "ATTRIBUTION.txt").write_text(
            "TinyStories by Ronen Eldan; license: CDLA-Sharing-1.0.\\n"
            "Dataset revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64.\\n"
            "https://huggingface.co/datasets/roneneldan/TinyStories\\n",
            encoding="utf-8",
        )
    else:
        for filename in ("train.txt", "valid.txt", "test.txt", "ATTRIBUTION.txt"):
            (out / filename).write_bytes((out / "source" / filename).read_bytes())
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=("language", "blender", "godot"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=3600)
    args = parser.parse_args()
    run_lane(args.lane, args.source, args.out, args.seconds)


if __name__ == "__main__":
    main()
