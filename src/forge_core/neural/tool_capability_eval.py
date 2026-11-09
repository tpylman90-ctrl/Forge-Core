"""Fresh-document and answer-keyed Blender/Godot probes for Forge checkpoints.

The benchmark scores next-token choice on original, paraphrased tasks derived
from official manual pages held out from specialist training. It does not prove
practical tool competence. Long-form code-generation tasks remain separate and
require human grading before any capability claim.
"""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import random
import time
import urllib.error
import urllib.request

import torch
from tokenizers import Tokenizer

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.multilane_v1 import SOURCES as TRAIN_SOURCES
from forge_core.neural.seed_v07 import heldout

SEQ_LEN = 128
DOC_WINDOWS = 128
LABELS = ("A", "B", "C", "D")

HOLDOUTS = {
    "blender": {
        "version": "Blender 5.4 Manual",
        "license": "CC BY-SA 4.0; prose only, code blocks excluded",
        "license_url": "https://docs.blender.org/manual/en/5.4/copyright.html",
        "attribution": "Blender Documentation Team, Blender 5.4 Manual",
        "urls": [
            "https://docs.blender.org/manual/en/5.4/modeling/meshes/editing/uv.html",
            "https://docs.blender.org/manual/en/5.4/modeling/meshes/uv/unwrapping/introduction.html",
            "https://docs.blender.org/manual/en/5.4/modeling/meshes/uv/editing.html",
            "https://docs.blender.org/manual/en/5.4/modeling/meshes/uv/unwrapping/seams.html",
            "https://docs.blender.org/manual/en/5.4/modeling/modifiers/introduction.html",
            "https://docs.blender.org/manual/en/5.4/modeling/meshes/editing/edge/bevel.html",
            "https://docs.blender.org/manual/en/5.4/modeling/modifiers/normals/weighted_normal.html",
            "https://docs.blender.org/manual/en/5.4/modeling/meshes/editing/mesh/normals.html",
        ],
    },
    "godot": {
        "version": "Godot 4.7 documentation",
        "license": "CC BY 3.0",
        "license_url": "https://docs.godotengine.org/en/4.7/about/faq.html",
        "attribution": "Juan Linietsky, Ariel Manzur, and the Godot Engine community",
        "urls": [
            "https://docs.godotengine.org/en/4.7/tutorials/physics/using_character_body_2d.html",
            "https://docs.godotengine.org/en/4.7/tutorials/scripting/gdscript/static_typing.html",
            "https://docs.godotengine.org/en/4.7/tutorials/scripting/gdscript/warning_system.html",
            "https://docs.godotengine.org/en/4.7/tutorials/physics/ray-casting.html",
            "https://docs.godotengine.org/en/4.7/classes/class_packedscene.html",
            "https://docs.godotengine.org/en/4.7/getting_started/step_by_step/signals.html",
            "https://docs.godotengine.org/en/4.7/classes/class_characterbody2d.html",
            "https://docs.godotengine.org/en/4.7/classes/class_node3d.html",
        ],
    },
}

# Each question is newly written. It maps to a held-out source page, while its
# four answer-order rotations balance letter-position preferences.
def _q(question_id, skill, situation, question, correct, distractors, source_index):
    return {
        "id": question_id, "skill": skill, "situation": situation,
        "question": question, "correct": correct,
        "distractors": distractors, "source_index": source_index,
    }

QUESTIONS = {
    "blender": [
        _q("uv-01", "unwrapping", "A mesh needs a 2D texture layout.",
           "What does UV unwrapping produce?", "A flattened 2D mapping of the mesh",
           ["A new 3D mesh with more vertices", "A material shader graph",
            "A set of object-space transforms"], 1),
        _q("uv-02", "unwrapping", "A textured model looks stretched after mapping.",
           "What does a UV map relate?", "Mesh surface points to locations in 2D texture space",
           ["Object origins to world axes", "Mesh objects to scene collections",
            "Materials to light sources"], 0),
        _q("uv-03", "unwrapping", "A model must be opened along chosen edges before flattening.",
           "What are seams used for?", "They mark where the surface can be cut for unwrapping",
           ["They weld nearby vertices", "They assign materials to faces",
            "They smooth vertex normals"], 3),
        _q("uv-04", "unwrapping", "An existing UV layout needs its cut lines restored.",
           "What does Seams from Islands do?", "Adds seams at boundaries of existing UV islands",
           ["Combines all islands into one face", "Deletes every UV coordinate",
            "Converts UVs into vertex colors"], 3),
        _q("uv-05", "layout", "Several UV islands need to use a texture efficiently.",
           "What is Pack Islands intended to do?", "Arrange islands to use texture space while respecting settings",
           ["Create new mesh faces", "Bake lighting into the mesh automatically",
            "Join separate objects into one"], 2),
        _q("uv-06", "layout", "A texture atlas requires a safety gap between islands.",
           "Which packing setting controls that gap?", "The island margin",
           ["The object origin", "The bevel width", "The scene frame rate"], 2),
        _q("uv-07", "editing", "Some UV points must stay fixed while nearby points are adjusted.",
           "What does pinning selected UVs provide?", "A way to keep selected UVs fixed during editing or unwrap",
           ["A way to lock the 3D object in world space", "A way to freeze the material",
            "A way to prevent mesh selection"], 2),
        _q("uv-08", "editing", "The layout is crowded and needs repositioning.",
           "Which operations can Pack Islands apply to UV islands?", "Scale, translate, and rotate them",
           ["Extrude, bevel, and subdivide them", "Move their mesh origins",
            "Change the object's parent"], 1),
        _q("uv-09", "unwrapping", "An artist marks cuts around a model before flattening its surface.",
           "How do seams affect unwrapping?", "They guide where the surface is split into UV regions",
           ["They define the model's rig", "They change the mesh's world scale",
            "They replace the material texture"], 3),
        _q("uv-10", "layout", "A lightmap layout should use as much texture area as practical.",
           "What tradeoff can a lightmap-focused UV layout make?", "It can prioritize dense texture use over a connected, easy-to-paint layout",
           ["It must remove all UV islands", "It preserves only object origins",
            "It disables texture sampling"], 0),
        _q("uv-11", "editing", "Faces were added after the mesh already had a UV map.",
           "What may be needed for the new faces?", "Additional UV mapping or editing",
           ["A new armature", "A new camera", "A scene-unit conversion"], 1),
        _q("uv-12", "unwrapping", "A model is being prepared for hand-painted texture work.",
           "Why is a useful UV layout important?", "It gives the 2D texture a readable mapping onto the 3D surface",
           ["It determines the object's physics mass", "It selects the active render engine",
            "It replaces mesh topology"], 0),
        _q("modifier-01", "modifier_stack", "A non-destructive effect needs to remain editable.",
           "What do Blender modifiers generally do before being applied?", "Change evaluated geometry while keeping the base mesh editable",
           ["Rewrite the base mesh immediately", "Convert the object into a collection",
            "Replace the scene camera"], 4),
        _q("modifier-02", "modifier_stack", "A Mirror and Subdivision modifier produce a different result when reordered.",
           "Why can changing modifier order change the result?", "Each modifier evaluates the output from earlier stack items",
           ["The stack is evaluated in random order", "Only the last modifier runs",
            "Modifier order changes the object's origin"], 4),
        _q("bevel-01", "beveling", "A bevel needs a smoother rounded edge.",
           "What does increasing bevel segments generally do?", "Adds denser geometry for a smoother bevel",
           ["Moves the object origin", "Removes the adjacent faces",
            "Changes the scene frame rate"], 5),
        _q("bevel-02", "beveling", "A bevel extends past narrow neighboring faces.",
           "Which bevel option helps prevent overshooting?", "Clamp Overlap",
           ["Loop Slide", "Pack Islands", "Object Origin"], 5),
        _q("normal-01", "shading_normals", "Hard-surface faces need modified shading normals.",
           "What does the Weighted Normal modifier change?", "The mesh's custom normals used for shading",
           ["The object's vertex positions", "The UV island layout",
            "The mesh's collection membership"], 6),
        _q("normal-02", "shading_normals", "A mesh has custom normals that affect its shading.",
           "What do mesh normals primarily describe?", "The direction used to represent surface orientation for shading",
           ["The object's world-space location", "The order of materials in a scene",
            "The pixels in a UV texture"], 7),
    ],
    "godot": [
        _q("body-01", "character_movement", "A player node must be moved by custom game logic.",
           "Which body is designed for code-controlled movement?", "CharacterBody2D or CharacterBody3D",
           ["StaticBody2D only", "WorldEnvironment", "AnimationLibrary"], 0),
        _q("body-02", "character_movement", "A character body should fall under gravity.",
           "Who calculates its gravity by default?", "The script must calculate and apply it",
           ["The physics engine always applies it automatically", "The renderer",
            "The scene importer"], 0),
        _q("body-03", "character_movement", "A character body is moved and must respond to collisions.",
           "Where should physics movement usually run?", "In _physics_process()",
           ["In _draw()", "In _ready() only", "In _input() only"], 3),
        _q("body-04", "character_movement", "A character should slide along a wall after colliding.",
           "Which movement method provides the common automatic slide response?", "move_and_slide()",
           ["set_global_position()", "move_and_collide() with no response code",
            "queue_redraw()"], 1),
        _q("body-05", "character_movement", "A projectile should stop at a collision so custom bounce logic can run.",
           "What does move_and_collide() return when it hits?", "A collision object with hit details",
           ["A new PackedScene", "A material resource", "A normalized input vector"], 2),
        _q("typing-01", "static_typing", "A variable should be explicitly declared as an integer.",
           "Which declaration uses a type hint?", "var health: int = 10",
           ["var health -> int = 10", "var int health = 10",
            "int var health = 10"], 1),
        _q("typing-02", "static_typing", "A variable should infer its type from its initial value.",
           "Which operator requests type inference in a variable declaration?", ":=",
           ["=>", "->", "::"], 2),
        _q("typing-03", "static_typing", "A list should contain integer scores.",
           "Which typed array form is supported?", "Array[int]",
           ["Array[Array[int]]", "int[Array]", "TypedArray[int, int]"], 0),
        _q("typing-04", "static_typing", "A team wants errors caught earlier and better editor completion.",
           "What can static typing help with?", "Finding more errors before running and improving completion",
           ["Automatically generating all scenes", "Replacing physics callbacks",
            "Guaranteeing every algorithm is correct"], 3),
        _q("warnings-01", "warnings", "A script has an unused-variable warning that is intentionally accepted.",
           "What annotation can ignore one named warning?", '@warning_ignore("unused_variable")',
           ["@warning_skip(unused_variable)", "# warning_ignore all",
            "disable_warning(unused_variable)"], 1),
        _q("warnings-02", "warnings", "A script has a warning in the editor.",
           "Do warnings stop the game from running by default?", "No; a warning can be configured as an error",
           ["Yes, every warning blocks the game", "Only warnings in comments block the game",
            "Warnings always delete the script"], 2),
        _q("ray-01", "ray_queries", "A query reads physics-space collision data.",
           "When is directly accessing the physics space safe?", "During _physics_process()",
           ["Only during _input()", "Only during _ready()", "Only during _draw()"], 0),
        _q("ray-02", "ray_queries", "A 2D ray query is sent between two global points.",
           "Which method performs the query on direct space state?", "intersect_ray(query)",
           ["move_and_slide(query)", "cast_to_global(query)", "get_node(query)"], 1),
        _q("ray-03", "ray_queries", "A ray hits the character that launched it.",
           "How can a query avoid hitting its own body?", "Put the body in the query's exclude list",
           ["Set the ray length to zero", "Disable all collision layers globally",
            "Move the character origin"], 1),
        _q("ray-04", "ray_queries", "A ray query returns no collision.",
           "What does the returned result contain?", "An empty dictionary",
           ["A null scene tree", "A new collider", "The full project settings"], 3),
        _q("scene-01", "packed_scenes", "A saved scene resource should create live nodes.",
           "Which PackedScene method creates the scene's node hierarchy?", "instantiate()",
           ["pack()", "get_state()", "queue_free()"], 4),
        _q("scene-02", "packed_scenes", "A script needs to test whether a scene resource contains nodes.",
           "What does PackedScene.can_instantiate() report?", "Whether the scene can be instantiated",
           ["Whether the scene is the main window", "Whether every child is visible",
            "Whether the resource is a physics body"], 4),
        _q("signal-01", "signals", "A reusable player scene should request a bullet spawn without choosing its parent.",
           "Why can a signal help here?", "It lets another connected node handle spawning without a fixed parent dependency",
           ["It automatically creates every node in the project", "It replaces PackedScene resources",
            "It applies gravity to the player"], 5),
        _q("signal-02", "signals", "A node declares an event that listeners should receive.",
           "How does GDScript emit a declared signal?", "Call the signal's emit() method",
           ["Call instantiate() on the signal", "Use queue_free() on the listener",
            "Write the signal into project settings"], 5),
        _q("physics-01", "physics_process", "A CharacterBody moves and must query collisions consistently.",
           "Where should its physics movement normally run?", "In _physics_process()",
           ["In _draw()", "Only in _ready()", "In a shader"], 6),
        _q("physics-02", "physics_process", "A CharacterBody should fall under gravity.",
           "Who applies gravity to a CharacterBody by default?", "The game script must apply gravity",
           ["The physics server always adds gravity automatically", "The renderer",
            "The scene importer"], 6),
        _q("node3d-01", "transforms", "A Node3D child is positioned relative to its parent.",
           "What does its local transform represent?", "Its transform relative to the parent coordinate system",
           ["Its position in screen pixels", "Its physics layer mask",
            "Its resource load order"], 7),
        _q("node3d-02", "transforms", "A local-space point must be converted to world space.",
           "Which Node3D method performs this conversion?", "to_global(local_point)",
           ["to_local(world_point)", "instantiate(local_point)", "add_child(local_point)"], 7),
    ],
}


class ProseHTMLParser(HTMLParser):
    DROP = {"script", "style", "noscript", "nav", "header", "footer",
            "aside", "pre", "code", "svg", "button"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self.DROP:
            self.hidden += 1
        elif not self.hidden and tag in {"p", "li", "h1", "h2", "h3", "dt", "dd", "br"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in self.DROP:
            self.hidden = max(0, self.hidden - 1)
        elif not self.hidden and tag in {"p", "li", "h1", "h2", "h3", "dt", "dd"}:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)

    def text(self) -> str:
        return " ".join(" ".join(self.parts).split())


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare_data(lane: str, out: Path) -> dict:
    if lane not in HOLDOUTS:
        raise ValueError("Only Blender and Godot tool lanes have fresh holdouts")
    spec = HOLDOUTS[lane]
    training_urls = set(TRAIN_SOURCES[lane]["urls"])
    if training_urls.intersection(spec["urls"]):
        raise ValueError("Fresh holdout pages overlap training pages")
    out.mkdir(parents=True, exist_ok=True)
    pages = []
    page_texts = []
    for i, url in enumerate(spec["urls"]):
        request = urllib.request.Request(url, headers={"User-Agent": "Forge-Core-Eval/1.0"})
        # The two lane jobs can fetch from the official documentation host at once.
        # Retry temporary rate limits and server errors with exponential backoff.
        for attempt in range(8):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    raw = response.read()
                    resolved = response.geturl()
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 429 and exc.code < 500:
                    raise
                if attempt == 7:
                    raise
                time.sleep(min(30, 2 ** attempt) + random.random())
            except (urllib.error.URLError, TimeoutError):
                if attempt == 7:
                    raise
                time.sleep(min(30, 2 ** attempt) + random.random())
        parser = ProseHTMLParser()
        parser.feed(raw.decode("utf-8", errors="replace"))
        text = parser.text()
        if len(text) < 1200:
            raise ValueError(f"Fresh holdout page too short: {url}")
        encoded = text.encode("utf-8")
        filename = f"page-{i + 1}.txt"
        (out / filename).write_bytes(encoded)
        page_texts.append(text)
        pages.append({
            "id": f"{lane}-fresh-{i + 1}", "url": url,
            "resolved_url": resolved, "filename": filename,
            "html_sha256": sha256(raw), "text_sha256": sha256(encoded),
            "characters": len(text),
        })
    combined = ("\n\n".join(page_texts) + "\n").encode("utf-8")
    manifest = {
        "schema": 1, "lane": lane, "version": spec["version"],
        "license": spec["license"], "license_url": spec["license_url"],
        "attribution": spec["attribution"],
        "code_samples": "HTML pre/code blocks excluded from extracted text",
        "role": "final evaluation only; these pages are excluded from all training and checkpoint selection",
        "split_policy": "all eight version-pinned topic pages are fresh held-out evaluation pages, never optimizer input",
        "pages": pages, "combined_text_sha256": sha256(combined),
        "question_bank_sha256": sha256(json.dumps(QUESTIONS[lane], sort_keys=True).encode()),
    }
    (out / "holdout-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / "ATTRIBUTION.txt").write_text(
        f"{spec['attribution']}\n{spec['version']}\n"
        f"License: {spec['license']}\n{spec['license_url']}\n"
        "Official manual prose is included with source attribution. Code blocks were excluded.\n",
        encoding="utf-8",
    )
    return manifest


def rotated_tasks(lane: str) -> list[dict]:
    if lane not in QUESTIONS:
        raise ValueError("Unknown lane")
    records = []
    for base in QUESTIONS[lane]:
        correct = base["correct"]
        distractors = base["distractors"]
        if correct in distractors or len(set([correct, *distractors])) != 4:
            raise ValueError(f"Question options must be unique: {base['id']}")
        for rotation in range(4):
            correct_pos = rotation
            options = [None] * 4
            options[correct_pos] = correct
            other_positions = [i for i in range(4) if i != correct_pos]
            for pos, choice in zip(other_positions, distractors):
                options[pos] = choice
            prompt = (
                f"Situation: {base['situation']}\nQuestion: {base['question']}\n"
                + "\n".join(f"{LABELS[i]}) {options[i]}" for i in range(4))
                + "\nAnswer:"
            )
            records.append({
                "id": f"{base['id']}:position-{LABELS[correct_pos]}",
                "base_id": base["id"], "skill": base["skill"],
                "source_url": HOLDOUTS[lane]["urls"][base["source_index"]],
                "prompt": prompt, "gold": LABELS[correct_pos],
                "options": options, "gold_text": correct,
            })
    return records


@torch.no_grad()
def score_tasks(model: ByteTransformer, tokenizer: Tokenizer, lane: str) -> dict:
    records = rotated_tasks(lane)
    results = []
    groups: dict[str, dict] = {}
    by_base: dict[str, list[bool]] = {}
    predicted_labels: dict[str, int] = {label: 0 for label in LABELS}
    label_predicted_counts: dict[str, int] = {label: 0 for label in LABELS}
    label_correct_by_base: dict[str, list[bool]] = {}
    label_correct = 0

    def continuation_score(prefix: list[int], text: str) -> float:
        candidate = tokenizer.encode(" " + text).ids
        if not candidate:
            raise ValueError("Choice text tokenized to an empty continuation")
        max_context = model.cfg.max_seq_len - len(candidate)
        if max_context < 1:
            raise ValueError("Choice text exceeds model context")
        context = prefix[-max_context:]
        input_ids = context + candidate[:-1]
        logits, _ = model(torch.tensor([input_ids], dtype=torch.long))
        target_logits = logits[0, len(context) - 1:len(context) - 1 + len(candidate)]
        target_ids = torch.tensor(candidate, dtype=torch.long)
        token_logprobs = torch.log_softmax(target_logits, dim=-1).gather(
            1, target_ids[:, None]
        ).squeeze(1)
        return float(token_logprobs.mean())

    for item in records:
        prefix = tokenizer.encode(item["prompt"]).ids
        if not prefix or len(prefix) >= model.cfg.max_seq_len:
            raise ValueError(f"Question exceeds model context: {item['id']}")
        scores = [continuation_score(prefix, choice) for choice in item["options"]]
        predicted_index = max(range(4), key=lambda i: scores[i])
        predicted = LABELS[predicted_index]
        predicted_labels[predicted] += 1
        correct = item["options"][predicted_index] == item["gold_text"]
        # Separately score the model's next answer label conditioned on the
        # complete question and option list. This exposes label/position bias
        # that ranking full answer text alone can hide.
        label_scores = [continuation_score(prefix, label) for label in LABELS]
        predicted_label_index = max(range(4), key=lambda i: label_scores[i])
        predicted_label = LABELS[predicted_label_index]
        label_is_correct = predicted_label == item["gold"]
        label_predicted_counts[predicted_label] += 1
        label_correct += int(label_is_correct)
        label_correct_by_base.setdefault(item["base_id"], []).append(label_is_correct)
        g = groups.setdefault(item["skill"], {"correct": 0, "total": 0})
        g["correct"] += int(correct)
        g["total"] += 1
        by_base.setdefault(item["base_id"], []).append(bool(correct))
        results.append({
            "id": item["id"], "base_id": item["base_id"], "skill": item["skill"],
            "gold": item["gold"], "predicted": predicted, "correct": bool(correct),
            "label_predicted": predicted_label, "label_correct": bool(label_is_correct),
            "label_mean_logprobs": {
                label: round(score, 6) for label, score in zip(LABELS, label_scores)
            },
            "choice_mean_logprobs": {
                label: round(score, 6) for label, score in zip(LABELS, scores)
            },
            "source_url": item["source_url"],
        })
    all_four = sum(len(outcomes) == 4 and all(outcomes) for outcomes in by_base.values())
    label_all_four = sum(
        len(outcomes) == 4 and all(outcomes)
        for outcomes in label_correct_by_base.values()
    )
    ncorrect = sum(x["correct"] for x in results)
    return {
        "correct": ncorrect, "total": len(results),
        "accuracy": round(ncorrect / len(results), 4),
        "base_question_count": len(by_base),
        "all_four_rotations_correct": all_four,
        "predicted_label_counts": predicted_labels,
        "by_skill": {
            k: {"correct": v["correct"], "total": v["total"],
                "accuracy": round(v["correct"] / v["total"], 4)}
            for k, v in sorted(groups.items())
        },
        "chance_baseline": 0.25,
        "label_selection_probe": {
            "correct": label_correct, "total": len(results),
            "accuracy": round(label_correct / len(results), 4),
            "base_question_count": len(label_correct_by_base),
            "all_four_rotations_correct": label_all_four,
            "predicted_label_counts": label_predicted_counts,
            "chance_baseline": 0.25,
            "scoring": "Rank the next-token log probability of A/B/C/D after the complete question and option list.",
        },
        "interpretation": "Full-answer-text ranking and answer-label selection are reported separately across four option rotations. Neither score certifies practical tool competence.",
        "items": results,
    }


@torch.no_grad()
def evaluate(checkpoint_dir: Path, data_dir: Path, out: Path) -> dict:
    manifest_path = data_dir / "holdout-manifest.json"
    if not manifest_path.exists():
        raise ValueError("Run --prepare-data first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lane = manifest["lane"]
    ckpt_path = checkpoint_dir / "checkpoint.pt"
    tok_path = checkpoint_dir / "tokenizer.json"
    if not ckpt_path.exists() or not tok_path.exists():
        raise ValueError("Checkpoint and tokenizer are required")
    ckpt_bytes = ckpt_path.read_bytes()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    tokenizer_sha = sha256(tok_path.read_bytes())
    if ckpt.get("tokenizer_sha256") != tokenizer_sha:
        raise ValueError("Checkpoint/tokenizer hash mismatch")
    tokenizer = Tokenizer.from_file(str(tok_path))
    model = ByteTransformer(ModelConfig(**ckpt["configuration"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    torch.set_num_threads(min(2, torch.get_num_threads()))
    fresh_pages = []
    for page in manifest["pages"]:
        data = (data_dir / page["filename"]).read_bytes()
        if sha256(data) != page["text_sha256"]:
            raise ValueError(f"Holdout text hash mismatch: {page['id']}")
        tokens = torch.tensor(tokenizer.encode(data.decode("utf-8")).ids, dtype=torch.long)
        if tokens.numel() <= SEQ_LEN + 1:
            raise ValueError(f"Holdout token page too short: {page['id']}")
        loss = heldout(model, tokens, windows=DOC_WINDOWS, seq_len=SEQ_LEN)
        fresh_pages.append({
            "id": page["id"], "url": page["url"], "text_sha256": page["text_sha256"],
            "token_count": int(tokens.numel()), "windows": DOC_WINDOWS,
            "token_loss": round(loss, 8),
        })
    task_scores = score_tasks(model, tokenizer, lane)
    training_metrics_path = checkpoint_dir / "metrics.json"
    training_metrics = None
    if training_metrics_path.exists():
        training_metrics = json.loads(training_metrics_path.read_text(encoding="utf-8"))
    report = {
        "schema": 1, "status": "evaluation_only_research",
        "lane": lane,
        "checkpoint_sha256": sha256(ckpt_bytes),
        "parent_checkpoint_sha256": ckpt.get("parent_sha256"),
        "parent_step": ckpt.get("parent_step"),
        "lane_updates": ckpt.get("lane_updates"),
        "seed": ckpt.get("seed"),
        "tokenizer_sha256": tokenizer_sha,
        "training_data_hashes": {k: ckpt.get(f"{k}_sha256") for k in ("train", "valid", "test")},
        "holdout_manifest_sha256": sha256(manifest_path.read_bytes()),
        "holdout_license": manifest["license"],
        "holdout_page_count": len(fresh_pages),
        "fresh_document_loss": round(sum(p["token_loss"] for p in fresh_pages) / len(fresh_pages), 8),
        "fresh_document_pages": fresh_pages,
        "capability_probe": task_scores,
        "task_samples_human_reviewed": False,
        "android_integrated": False,
        "promoted": False,
        "training_metrics": training_metrics,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=("blender", "godot"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.prepare_data:
        manifest = prepare_data(args.lane, args.data_dir)
        print(json.dumps({"prepared": args.lane, "manifest": manifest}, indent=2))
        return
    if args.checkpoint_dir is None or args.out is None:
        parser.error("--checkpoint-dir and --out are required unless --prepare-data is set")
    report = evaluate(args.checkpoint_dir, args.data_dir, args.out)
    print(json.dumps({
        "lane": report["lane"], "checkpoint_sha256": report["checkpoint_sha256"],
        "fresh_document_loss": report["fresh_document_loss"],
        "task_correct": report["capability_probe"]["correct"],
        "task_total": report["capability_probe"]["total"],
    }, indent=2))


if __name__ == "__main__":
    main()
