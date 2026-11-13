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
import urllib.error
import urllib.request

import torch
from tokenizers import Tokenizer

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.seed_v07 import heldout
from forge_core.neural.train import make_batch

BASE_RUN = 42552572467
BASE_SHA256 = "848553a81549c1269c360c5d07c57adde92a7816a0bc19b61ecc8028c93fee99"
BASE_STEP = 107250
BASE_TOKENIZER_SHA256 = "4ecd7803d4c369e80bda2e74c30cfb769b7e3a942f931df83383eec9f794f04c"
MAX_SECONDS = 10800
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
            "https://docs.blender.org/manual/en/5.4/modeling/meshes/structure.html",
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

# Original Forge-Core lesson cards. Authored for this project and released as CC0-1.0.
# Each card asks for an action, provides a worked answer, and explains why it works.
LESSONS = {
    "blender": [
        {
            "task": "Build a tapered low-poly rock mesh from explicit vertices and faces.",
            "answer": "import bpy\nvertices = [(-1, -0.7, 0), (0.8, -0.8, 0), (1, 0.6, 0), (-0.7, 0.9, 0), (-0.3, -0.2, 1.2), (0.25, 0.35, 1.0)]\nfaces = [(0, 1, 4), (1, 2, 5, 4), (2, 3, 5), (3, 0, 4, 5), (0, 3, 2, 1)]\nmesh = bpy.data.meshes.new('Rock_Mesh')\nmesh.from_pydata(vertices, [], faces)\nmesh.update()\nrock = bpy.data.objects.new('Low_Poly_Rock', mesh)\nbpy.context.collection.objects.link(rock)",
            "why": "Mesh.from_pydata takes vertex coordinates, optional edges, and face index lists. Creating the object separately keeps mesh data and scene object identity clear; the object must be linked into a collection to appear in the scene."
        },
        {
            "task": "Create a two-unit cube and add a three-segment bevel modifier.",
            "answer": "import bpy\nbpy.ops.mesh.primitive_cube_add(size=2)\nobj = bpy.context.object\nobj.name = 'Beveled_Block'\nbpy.ops.object.transform_apply(location=False, rotation=False, scale=True)\nbevel = obj.modifiers.new('Edge_Bevel', 'BEVEL')\nbevel.width = 0.08\nbevel.segments = 3",
            "why": "Applying scale before setting bevel width helps keep the edge width consistent when the object was non-uniformly scaled. A modifier keeps the edit non-destructive until it is applied."
        },
        {
            "task": "Set the active mesh object's origin to its geometry bounds center.",
            "answer": "import bpy\nobj = bpy.context.active_object\nif obj is None or obj.type != 'MESH':\n    raise RuntimeError('Select a mesh object first')\nbpy.context.view_layer.objects.active = obj\nobj.select_set(True)\nbpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')",
            "why": "The origin is the object transform pivot. Checking the active object and type before using an operator makes the script fail clearly when its precondition is missing."
        },
        {
            "task": "Export only the active mesh as a binary GLB file.",
            "answer": "import bpy\nobj = bpy.context.active_object\nif obj is None or obj.type != 'MESH':\n    raise RuntimeError('Select a mesh object first')\nbpy.ops.object.select_all(action='DESELECT')\nobj.select_set(True)\nbpy.context.view_layer.objects.active = obj\nbpy.ops.export_scene.gltf(filepath='asset.glb', export_format='GLB', use_selection=True)",
            "why": "Restricting the exporter to the intended selection prevents cameras, lights, and unrelated scene objects from entering the asset. Inspect the exported file in the target engine as a separate check."
        },
        {
            "task": "Add a named material and assign a blue viewport color to an active mesh.",
            "answer": "import bpy\nobj = bpy.context.active_object\nif obj is None or obj.type != 'MESH':\n    raise RuntimeError('Select a mesh object first')\nmat = bpy.data.materials.new('Enamel_Blue')\nmat.diffuse_color = (0.08, 0.25, 0.8, 1.0)\nobj.data.materials.append(mat)",
            "why": "A material datablock can be created independently and assigned through the mesh material slots. The diffuse color is a four-component RGBA value used for viewport display and related material settings."
        },
        {
            "task": "Add a weighted-normal modifier after creating hard-surface bevels.",
            "answer": "import bpy\nobj = bpy.context.active_object\nif obj is None or obj.type != 'MESH':\n    raise RuntimeError('Select a mesh object first')\nobj.modifiers.new('Weighted_Normals', 'WEIGHTED_NORMAL')",
            "why": "A weighted-normal modifier changes shading normals to influence how the surface is lit; it does not add bevel geometry. Modifier order should be reviewed because the stack is evaluated in order."
        },
        {
            "task": "Create and link a named collection, then move the active object into it.",
            "answer": "import bpy\nobj = bpy.context.active_object\nif obj is None:\n    raise RuntimeError('Select an object first')\ncollection = bpy.data.collections.new('Generated_Props')\nbpy.context.scene.collection.children.link(collection)\nfor old_collection in list(obj.users_collection):\n    old_collection.objects.unlink(obj)\ncollection.objects.link(obj)",
            "why": "Collections organize scene objects. An object must be unlinked from its old collections before being moved when the goal is exclusive membership."
        },
        {
            "task": "Make a duplicate object with an independent copy of its mesh data.",
            "answer": "import bpy\nsource = bpy.context.active_object\nif source is None or source.type != 'MESH':\n    raise RuntimeError('Select a mesh object first')\nduplicate = source.copy()\nduplicate.data = source.data.copy()\nduplicate.name = source.name + '_Independent'\nbpy.context.collection.objects.link(duplicate)",
            "why": "Copying an object alone can leave it sharing the original mesh datablock. Copying both object and mesh data allows later mesh edits to remain independent."
        },
    ],
    "godot": [
        {
            "task": "Load a PackedScene, instantiate it, and add the instance under the current node.",
            "answer": "var actor_scene: PackedScene = preload('res://actors/actor.tscn')\nvar actor: Node = actor_scene.instantiate()\nadd_child(actor)",
            "why": "A PackedScene is a reusable scene resource. instantiate() creates a live node hierarchy, and add_child() inserts it into the active tree under the intended parent."
        },
        {
            "task": "Declare and emit a typed health_changed signal when health is updated.",
            "answer": "signal health_changed(current_health: int)\nvar health: int = 100\n\nfunc set_health(value: int) -> void:\n    health = value\n    health_changed.emit(health)",
            "why": "A signal reports an event to connected listeners without requiring the emitting node to know which node will respond. A typed argument documents the value passed to listeners."
        },
        {
            "task": "Move a CharacterBody3D using input, gravity, and physics-step movement.",
            "answer": "extends CharacterBody3D\n@export var speed: float = 5.0\n@export var gravity: float = 9.8\n\nfunc _physics_process(delta: float) -> void:\n    var input := Input.get_vector('move_left', 'move_right', 'move_forward', 'move_back')\n    velocity.x = input.x * speed\n    velocity.z = input.y * speed\n    if not is_on_floor():\n        velocity.y -= gravity * delta\n    move_and_slide()",
            "why": "CharacterBody movement is scripted in the physics callback. Gravity is applied by the script, and move_and_slide() performs collision-aware movement using the body velocity."
        },
        {
            "task": "Build one triangle surface with ArrayMesh from three vertices and indices.",
            "answer": "var arrays := []\narrays.resize(Mesh.ARRAY_MAX)\narrays[Mesh.ARRAY_VERTEX] = PackedVector3Array([Vector3.ZERO, Vector3.RIGHT, Vector3.UP])\narrays[Mesh.ARRAY_INDEX] = PackedInt32Array([0, 1, 2])\nvar mesh := ArrayMesh.new()\nmesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)",
            "why": "ArrayMesh receives vertex attributes in indexed slots. The index list selects the three vertices that form a triangle; extra arrays can be supplied for normals, UVs, and other attributes."
        },
        {
            "task": "Convert a local point on a Node3D into world coordinates.",
            "answer": "var world_point: Vector3 = node.to_global(local_point)",
            "why": "A Node3D transform is relative to its parent. to_global() applies the node's transform hierarchy to convert a local-space point to world space."
        },
        {
            "task": "Connect a button's pressed signal to a named callback.",
            "answer": "func _ready() -> void:\n    $Button.pressed.connect(_on_button_pressed)\n\nfunc _on_button_pressed() -> void:\n    print('Button pressed')",
            "why": "Connecting during setup gives the event a clear receiver. A signal lets the button report an event without owning the behavior that responds to it."
        },
        {
            "task": "Load a PackedScene from a runtime path, check the result, and instantiate it.",
            "answer": "var packed: PackedScene = load(scene_path) as PackedScene\nif packed == null:\n    push_error('Could not load scene: ' + scene_path)\n    return\nvar instance := packed.instantiate()\nadd_child(instance)",
            "why": "load() is useful when a resource path is selected at runtime. Checking the cast before calling instantiate() handles a missing resource or a path that points to the wrong resource type."
        },
        {
            "task": "Delete a node safely after the current frame's scene operations.",
            "answer": "queue_free()",
            "why": "queue_free() schedules a node for deletion at the end of the frame, avoiding immediate destruction while the scene tree may still be processing it."
        },
    ],
}

TASK_PROMPTS = {
    "blender": [
        "Write a Blender Python script that creates a tapered low-poly rock mesh from vertices and faces.",
        "Create a hard-surface cube with applied scale, a bevel modifier, and a clear object name.",
        "Explain how object origin placement changes later rotation and placement.",
        "Write a safe Blender script that exports only the active mesh to a GLB.",
        "Create and assign a named material to the active mesh, including a viewport color.",
        "Explain why a modifier stack's order can change the final mesh.",
        "Write a script that moves the active object into a newly created collection.",
        "Duplicate a mesh object so the duplicate has independent mesh data.",
    ],
    "godot": [
        "Write GDScript that loads and instantiates a PackedScene, reports a missing resource, and adds the node.",
        "Declare a typed signal and emit it when a health value changes.",
        "Write CharacterBody3D movement using physics input, gravity, and move_and_slide.",
        "Create a triangle ArrayMesh from vertex and index arrays.",
        "Explain local and global coordinates and convert a local Node3D point to world space.",
        "Connect a Button.pressed signal to a named callback.",
        "Explain why a signal can make a reusable scene less dependent on its parent.",
        "Explain when queue_free is safer than immediate node deletion.",
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
    manifest_path = out / "source-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("lane") != lane or manifest.get("version") != config["version"]:
            raise ValueError("Cached manual corpus lane/version mismatch")
        expected_urls = [item["url"] for item in manifest.get("pages", [])]
        if expected_urls != config["urls"]:
            raise ValueError("Cached manual corpus URL set mismatch")
        for split in ("train", "valid", "test"):
            if sha256_file(out / f"{split}.txt") != manifest.get(f"{split}_sha256"):
                raise ValueError(f"Cached manual {split} hash mismatch")
        lesson_text = "\n\n".join(
            f"Task: {item['task']}\nAnswer:\n{item['answer']}\nWhy:\n{item['why']}"
            for item in LESSONS[lane]
        )
        if sha256_bytes(lesson_text.encode("utf-8")) != manifest.get("lesson_sha256"):
            raise ValueError("Cached teacher lesson cards differ from the current source")
        return manifest

    pages, texts = [], []
    for url in config["urls"]:
        request = urllib.request.Request(url, headers={"User-Agent": "Forge-Core-Research/1.0"})
        for attempt in range(8):
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    raw, final_url = response.read(), response.geturl()
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 429 and exc.code < 500:
                    raise
                if attempt == 7:
                    raise
                time.sleep(min(30, 2 ** attempt) + random.random())
            except urllib.error.URLError:
                if attempt == 7:
                    raise
                time.sleep(min(30, 2 ** attempt) + random.random())
        parser = ManualText()
        parser.feed(raw.decode("utf-8", errors="replace"))
        text = parser.text()
        if len(text) < 1500:
            raise ValueError(f"Manual page yielded too little prose: {url}")
        pages.append({"url": url, "resolved_url": final_url,
                      "html_sha256": sha256_bytes(raw), "extracted_characters": len(text)})
        texts.append(text)
    lesson_text = "\n\n".join(
        f"Task: {item['task']}\nAnswer:\n{item['answer']}\nWhy:\n{item['why']}"
        for item in LESSONS[lane]
    )
    lesson_bytes = lesson_text.encode("utf-8")
    train_text = "\n\n".join(texts[:4]) + "\n\n" + lesson_text
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
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "ATTRIBUTION.txt").write_text(
        f"{config['attribution']}\n{config['version']}\n"
        f"License: {config['license']}\n{config['license_url']}\n"
        "The extracted source text is provided only with this attribution and license notice.\n",
        encoding="utf-8",
    )
    return manifest

def fetch_language_test(source: Path, manifest: dict, byte_offset: int = 0,
                        sample_bytes: int = 4 * 1024 * 1024,
                        exclude_paths: tuple[Path, ...] = ()) -> tuple[str, dict]:
    """Fetch a fixed TinyStories byte range beyond the trained prefix.

    byte_offset selects a later non-training range; sample_bytes must be a
    positive multiple of 4 MiB. Optional prior test files are excluded exactly.
    """
    from forge_core.neural.pinned_corpus import clean_prefix, complete_stories, story_hash

    train_limit = int(manifest["source_train_bytes_requested"])
    block_size = 4 * 1024 * 1024
    if byte_offset < 0 or byte_offset % block_size:
        raise ValueError("Test offset must be a nonnegative multiple of 4 MiB")
    if sample_bytes < block_size or sample_bytes % block_size:
        raise ValueError("Test sample size must be a positive multiple of 4 MiB")
    start = train_limit + byte_offset
    end = start + sample_bytes - 1
    request = urllib.request.Request(
        manifest["train_url"],
        headers={
            "User-Agent": "Forge-Core-audit/1.0 (pinned public corpus)",
            "Range": f"bytes={start}-{end}",
        },
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = response.read(sample_bytes)
                content_range = response.headers.get("Content-Range", "")
                status = response.status
            break
        except urllib.error.HTTPError as exc:
            if (exc.code != 429 and exc.code < 500) or attempt == 4:
                raise
            time.sleep(min(16, 2 ** attempt) + random.random())
        except (urllib.error.URLError, TimeoutError):
            if attempt == 4:
                raise
            time.sleep(min(16, 2 ** attempt) + random.random())
    if status != 206 or not content_range.startswith(f"bytes {start}-"):
        raise ValueError("TinyStories server did not honor the independent byte-range request")
    if len(raw) != sample_bytes:
        raise ValueError("Independent TinyStories range was shorter than requested")

    decoded = raw.decode("utf-8", errors="ignore").replace("\r\n", "\n")
    first_boundary = decoded.find("\n\n")
    last_boundary = decoded.rfind("\n\n")
    if first_boundary < 0 or last_boundary <= first_boundary:
        raise ValueError("Independent range did not contain complete TinyStories records")
    complete_range = decoded[first_boundary + 2:last_boundary + 2].encode("utf-8")

    train_text = (source / "train.txt").read_text(encoding="utf-8")
    valid_text = (source / "valid.txt").read_text(encoding="utf-8")
    blocked = {story_hash(item) for item in complete_stories(
        (train_text.rstrip() + "\n\n" + valid_text).encode("utf-8")
    )}
    for prior_path in exclude_paths:
        if prior_path.exists():
            blocked.update(story_hash(item) for item in complete_stories(
                prior_path.read_bytes()
            ))
    before = len(complete_stories(complete_range))
    test_text, document_count = clean_prefix(complete_range, blocked=blocked)
    if not test_text.strip():
        raise ValueError("Independent language test sample is empty")
    test_path = source / "test.txt"
    test_path.write_text(test_text, encoding="utf-8")
    test_sha = sha256_file(test_path)
    if test_sha in {manifest["train_sha256"], manifest["valid_sha256"]}:
        raise ValueError("Language test data duplicates a train or validation split")
    enriched = dict(manifest)
    enriched["test_sha256"] = test_sha
    enriched["test_document_count"] = document_count
    enriched["test_utf8_bytes"] = test_path.stat().st_size
    enriched["test_source"] = {
        "url": manifest["train_url"],
        "license": manifest["license"],
        "revision": manifest["revision"],
        "range_start_inclusive": start,
        "range_end_inclusive": end,
        "byte_offset_after_training_prefix": byte_offset,
        "sample_bytes": sample_bytes,
        "raw_range_sha256": sha256_bytes(raw),
        "complete_records_before_duplicate_filter": before,
        "exact_duplicates_removed": before - document_count,
        "prior_test_files_excluded": [p.name for p in exclude_paths if p.exists()],
        "policy": "pinned training prefix followed by a separate fixed byte-range reserved for evaluation; byte_offset_after_training_prefix records skipped ranges",
    }
    return test_text, enriched


def load_language_data(source: Path, test_offset_bytes: int = 0) -> tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    manifest = json.loads((source / "source-manifest.json").read_text())
    ckpt = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    for split in ("train", "valid"):
        actual = sha256_file(source / f"{split}.txt")
        assert actual == manifest[f"{split}_sha256"] == ckpt[f"{split}_sha256"]
    tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    train = torch.tensor(tokenizer.encode((source / "train.txt").read_text()).ids, dtype=torch.long)
    valid = torch.tensor(tokenizer.encode((source / "valid.txt").read_text()).ids, dtype=torch.long)
    test_text, manifest = fetch_language_test(source, manifest, byte_offset=test_offset_bytes)
    test = torch.tensor(tokenizer.encode(test_text).ids, dtype=torch.long)
    if manifest["test_sha256"] == manifest["valid_sha256"]:
        raise ValueError("Language test and validation hashes must be distinct")
    return sha256_file(source / "tokenizer.json"), train, valid, test, manifest


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
        "schema": 3, "lane": lane, "parent_run": BASE_RUN,
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

@torch.no_grad()
def generate_task_samples(model, tokenizer: Tokenizer, lane: str) -> list[dict]:
    samples = []
    was_training = model.training
    model.eval()
    with torch.random.fork_rng():
        for index, prompt in enumerate(TASK_PROMPTS[lane]):
            torch.manual_seed(12000 + index)
            prefix = tokenizer.encode(prompt + "\nAnswer:\n").ids
            generated = model.generate(
                torch.tensor([prefix], dtype=torch.long), new_tokens=96, temperature=0.35
            )
            response = tokenizer.decode(generated[0].tolist()[len(prefix):],
                                        skip_special_tokens=True)
            samples.append({
                "prompt": prompt, "response": response,
                "human_review": "not_reviewed",
            })
    model.train(was_training)
    return samples


def run_lane(lane: str, source: Path, out: Path, seconds: int, max_updates: int | None = None,
             seed_override: int | None = None, learning_rate_override: float | None = None,
             eval_every: int = 1000, patience: int = 0):
    if lane not in {"language", "blender", "godot"}:
        raise ValueError("Unknown lane")
    if not 60 <= seconds <= MAX_SECONDS:
        raise ValueError("Each isolated lane must run from 60 to 3,600 seconds")
    if max_updates is not None and not 1 <= max_updates <= 40000:
        raise ValueError("max_updates must be between 1 and 40000")
    if eval_every < 1 or eval_every > 10000:
        raise ValueError("eval_every must be between 1 and 10000")
    if patience < 0 or patience > 20:
        raise ValueError("patience must be between 0 and 20")
    if seed_override is not None and not 1 <= seed_override <= 2147483647:
        raise ValueError("seed must be a positive 32-bit integer")
    if learning_rate_override is not None and not 0 < learning_rate_override <= 0.01:
        raise ValueError("learning_rate must be in (0, 0.01]")
    parent_path = source / "checkpoint.pt"
    parent_hash = sha256_file(parent_path)
    if parent_hash != BASE_SHA256:
        raise ValueError(f"Wrong Forge parent checkpoint: {parent_hash}")
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    if parent.get("step") != BASE_STEP or parent.get("seed") != 43:
        raise ValueError("Expected the exact verified selected 107250-step Forge Seed checkpoint")
    if sha256_file(source / "tokenizer.json") != BASE_TOKENIZER_SHA256:
        raise ValueError("Base tokenizer hash mismatch")
    tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    if lane == "language":
        _, train_ids, valid_ids, test_ids, manifest = load_language_data(source)
        core_valid_ids = valid_ids
    else:
        train_ids, valid_ids, test_ids, manifest = load_manual_data(lane, out / "source", tokenizer)
        core_valid_ids = torch.tensor(
            tokenizer.encode((source / "valid.txt").read_text(encoding="utf-8")).ids,
            dtype=torch.long,
        )

    seed = seed_override if seed_override is not None else (43 if lane == "language" else (731 if lane == "blender" else 947))
    eval_windows = 2048 if lane == "language" else EVAL_WINDOWS
    torch.set_num_threads(min(2, torch.get_num_threads()))
    torch.manual_seed(seed)
    random.seed(seed)
    model = ByteTransformer(ModelConfig(**parent["configuration"]))
    model.load_state_dict(parent["model"])
    learning_rate = learning_rate_override if learning_rate_override is not None else (
        float(parent.get("lr", 3e-4)) if lane == "language" else 5e-5
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    if lane == "language":
        optimizer.load_state_dict(parent["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        torch.set_rng_state(parent["torch_rng_state"])
        rng = torch.Generator()
        rng.set_state(parent["sample_rng_state"])
    else:
        rng = torch.Generator().manual_seed(seed + 1)

    out.mkdir(parents=True, exist_ok=True)
    model.train()
    initial_valid = heldout(model, valid_ids, windows=eval_windows, seq_len=SEQ_LEN)
    initial_test = heldout(model, test_ids, windows=eval_windows, seq_len=SEQ_LEN)
    initial_core_language_loss = heldout(
        model, core_valid_ids, windows=2048, seq_len=SEQ_LEN
    )
    best_valid = initial_valid
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_optimizer = copy.deepcopy(optimizer.state_dict())
    best_torch_rng = torch.get_rng_state().clone()
    best_sample_rng = rng.get_state().clone()
    started = time.monotonic()
    updates = 0
    curve = []
    stale_evaluations = 0
    stop_reason = "time_budget"
    while time.monotonic() - started < seconds and (max_updates is None or updates < max_updates):
        x, y = make_batch(train_ids, BATCH, SEQ_LEN, rng)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        updates += 1
        if updates % eval_every == 0 or (max_updates is not None and updates == max_updates):
            score = heldout(model, valid_ids, windows=eval_windows, seq_len=SEQ_LEN)
            curve.append({"updates": updates, "valid_token_loss": round(score, 8)})
            improved = lane == "language" or score < best_valid
            if improved:
                best_valid = score
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_optimizer = copy.deepcopy(optimizer.state_dict())
                best_torch_rng = torch.get_rng_state().clone()
                best_sample_rng = rng.get_state().clone()
                stale_evaluations = 0
            elif lane != "language":
                stale_evaluations += 1
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
                              "stale_evaluations": stale_evaluations,
                              "elapsed_seconds": round(time.monotonic() - started, 1)}),
                  flush=True)
            if lane != "language" and patience and stale_evaluations >= patience:
                stop_reason = "early_stopping"
                break
    if max_updates is not None and updates >= max_updates:
        stop_reason = "max_updates"
    if lane == "language":
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        best_optimizer = copy.deepcopy(optimizer.state_dict())
        best_torch_rng = torch.get_rng_state().clone()
        best_sample_rng = rng.get_state().clone()
    model.load_state_dict(best_state)
    selected_valid = heldout(model, valid_ids, windows=eval_windows, seq_len=SEQ_LEN)
    final_test = heldout(model, test_ids, windows=eval_windows, seq_len=SEQ_LEN)
    selected_core_language_loss = heldout(
        model, core_valid_ids, windows=2048, seq_len=SEQ_LEN
    )
    final_optimizer = optimizer.state_dict() if lane == "language" else best_optimizer
    save_checkpoint(out, model, final_optimizer, lane, parent, updates, seed,
                    {"rng": rng, "manifest": manifest}, best_valid,
                    best_torch_rng, best_sample_rng)
    task_samples = generate_task_samples(model, tokenizer, lane) if lane in TASK_PROMPTS else []
    if task_samples:
        (out / "task-samples.json").write_text(json.dumps(task_samples, indent=2) + "\n")
    report = {
        "schema": 1, "status": "research_only_not_promoted", "lane": lane,
        "parent_run": BASE_RUN, "parent_checkpoint_sha256": parent_hash,
        "parent_step": BASE_STEP, "candidate_checkpoint_sha256": sha256_file(out / "checkpoint.pt"),
        "candidate_language_step": BASE_STEP + updates if lane == "language" else None,
        "updates": updates, "elapsed_seconds": round(time.monotonic() - started, 2),
        "parameters": model.num_parameters, "tokenizer_sha256": BASE_TOKENIZER_SHA256,
        "train_sha256": manifest["train_sha256"], "valid_sha256": manifest["valid_sha256"],
        "test_sha256": manifest["test_sha256"],
        "validation_windows": eval_windows,
        "validation_loss_parent": round(initial_valid, 8),
        "validation_loss_selected": round(selected_valid, 8),
        "test_loss_parent": round(initial_test, 8), "test_loss_candidate": round(final_test, 8),
        "curve": curve,
        "core_language_validation": {
            "split": "shared pinned TinyStories validation set",
            "windows": 2048,
            "parent_loss": round(initial_core_language_loss, 8),
            "candidate_loss": round(selected_core_language_loss, 8),
            "delta_candidate_minus_parent": round(selected_core_language_loss - initial_core_language_loss, 8),
            "lower_is_better": True,
            "interpretation": "Matched cross-lane retention check; it is a text-prediction diagnostic, not proof of general language ability.",
        },
        "training_config": {"learning_rate": learning_rate, "eval_every": eval_every,
                            "patience": patience, "stop_reason": stop_reason,
                            "seed": seed},
        "capability_claim": "None. Documentation token loss and ungraded generated samples are not proof of tool competence.",
        "task_samples_human_reviewed": False,
        "android_integrated": False,
        "optimizer_policy": "resumed only for language; fresh and lane-local for specialist branches",
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "lineage.json").write_text(json.dumps({
        "lane": lane, "parent_run": BASE_RUN, "parent_step": BASE_STEP,
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
        (out / "test.txt").write_bytes((source / "test.txt").read_bytes())
        (out / "ATTRIBUTION.txt").write_text(
            "TinyStories by Ronen Eldan; license: CDLA-Sharing-1.0.\n"
            "Dataset revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64.\n"
            "https://huggingface.co/datasets/roneneldan/TinyStories\n",
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
    parser.add_argument("--max-updates", type=int, help="Optional bounded update count")
    parser.add_argument("--seed", type=int, help="Override lane's reproducible seed")
    parser.add_argument("--learning-rate", type=float, help="Override the fresh optimizer learning rate")
    parser.add_argument("--eval-every", type=int, default=1000, help="Updates between validation checkpoints")
    parser.add_argument("--patience", type=int, default=0, help="Specialist validation evaluations without improvement before stopping")
    args = parser.parse_args()
    run_lane(args.lane, args.source, args.out, args.seconds, args.max_updates,
             args.seed, args.learning_rate, args.eval_every, args.patience)


if __name__ == "__main__":
    main()
