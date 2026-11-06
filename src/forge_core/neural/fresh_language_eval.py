"""Fresh, deterministic mirrored language probe for Forge Seed candidates.

This is a small task diagnostic, not a standardized comprehension benchmark.
Its seed and generated items are recorded so each matched arm sees the same
held-out questions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from forge_core.neural.model import ByteTransformer, ModelConfig

NAMES = ("Ava", "Ezra", "Nia", "Omar", "Lena", "Jules", "Priya", "Theo",
         "Maya", "Rafi", "Iris", "Dante", "Sana", "Milo", "Tess", "Arlo")
COLORS = ("amber", "teal", "coral", "silver", "violet", "maroon", "ivory", "navy")
OBJECTS = ("shell", "scarf", "lantern", "button", "marble", "basket", "feather", "ribbon")
PLACES = ("bench", "crate", "stool", "shelf", "arch", "bridge", "porch", "cabin")
RELATIONS = ("behind", "beneath", "beside", "inside", "above", "outside", "between", "near")
NUMBERS = (("two", "five"), ("three", "six"), ("four", "seven"), ("one", "eight"),
           ("six", "nine"), ("five", "ten"), ("two", "seven"), ("three", "eight"))


def build_tasks(seed: int = 941027) -> list[dict]:
    rng = random.Random(seed)
    tasks: list[dict] = []
    for i in range(20):
        a, b = rng.sample(NAMES, 2)
        skill = ("color", "recipient", "count", "location", "agent")[i % 5]
        if skill == "color":
            target, other = rng.sample(COLORS, 2)
            obj, distractor = rng.sample(OBJECTS, 2)
            story = f"{a} carried a {target} {obj}. {b} carried a {other} {distractor}."
            question, correct, wrong = f"What color was {a}'s {obj}?", target, other
        elif skill == "recipient":
            obj = rng.choice(OBJECTS)
            story = f"{a} passed the {obj} to {b} before the bell rang."
            question, correct, wrong = "Who received the object?", b, a
        elif skill == "count":
            first, second = rng.choice(NUMBERS)
            thing = rng.choice(("smooth stones", "paper stars", "wooden beads", "small cups"))
            story = f"{a} counted {first} {thing}; {b} counted {second} {thing}."
            question, correct, wrong = f"How many {thing} did {a} count?", first, second
        elif skill == "location":
            relation, decoy = rng.sample(RELATIONS, 2)
            obj, place = rng.sample(OBJECTS, 2)
            landmark = rng.choice(PLACES)
            story = f"{a} set the {obj} {relation} the {landmark}. {b} put a {place} {decoy} the {landmark}."
            question, correct, wrong = f"Where did {a} set the {obj}?", relation, decoy
        else:
            obj = rng.choice(OBJECTS)
            action, other_action = rng.sample(("mended", "carried", "folded", "painted", "washed", "opened"), 2)
            story = f"{a} {action} the {obj} while {b} watched from the doorway."
            question, correct, wrong = f"Who {action} the {obj}?", a, b
        tasks.append({"id": f"fresh-{i:02d}", "skill": skill, "story": story,
                      "question": question, "correct": correct, "distractor": wrong})
    return tasks


def make_prompts(tasks: list[dict]) -> list[dict]:
    records = []
    for item in tasks:
        for flip in (0, 1):
            left, right = ((item["correct"], item["distractor"]) if flip == 0
                           else (item["distractor"], item["correct"]))
            prompt = (f"Read the scene carefully. {item['story']}\n"
                      f"Question: {item['question']}\n"
                      f"A) {left}\nB) {right}\nAnswer:")
            records.append({"id": f"{item['id']}-mirror{flip}", "skill": item["skill"],
                            "prompt": prompt, "gold": "A" if flip == 0 else "B"})
    return records


@torch.no_grad()
def score(model: ByteTransformer, tokenizer: Tokenizer, prompts: list[dict]) -> dict:
    model.eval()
    groups: dict[str, dict] = {}
    outcomes = []
    for task in prompts:
        prefix = tokenizer.encode(task["prompt"]).ids
        if len(prefix) >= model.cfg.max_seq_len:
            raise ValueError("Fresh probe exceeds the model context window")
        logits, _ = model(torch.tensor([prefix], dtype=torch.long))
        probabilities = F.log_softmax(logits[0, -1], dim=0)
        a_id, b_id = tokenizer.encode("A").ids[0], tokenizer.encode("B").ids[0]
        predicted = "A" if float(probabilities[a_id]) >= float(probabilities[b_id]) else "B"
        correct = int(predicted == task["gold"])
        group = groups.setdefault(task["skill"], {"correct": 0, "total": 0})
        group["correct"] += correct
        group["total"] += 1
        outcomes.append(correct)
    pairs = [(outcomes[i], outcomes[i + 1]) for i in range(0, len(outcomes), 2)]
    return {
        "correct": sum(outcomes), "total": len(outcomes),
        "mirrored_pairs_both_correct": sum(a and b for a, b in pairs),
        "mirrored_pair_count": len(pairs),
        "by_skill": {k: {**v, "accuracy": round(v["correct"] / v["total"], 4)}
                     for k, v in sorted(groups.items())},
        "limit": "20 newly generated factual toy scenarios, forced A/B first-token scoring; not proof of comprehension.",
    }


def evaluate(folder: Path, out: Path, seed: int = 941027) -> dict:
    checkpoint_path = folder / "checkpoint.pt"
    tokenizer_path = folder / "tokenizer.json"
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model = ByteTransformer(ModelConfig(**ckpt["configuration"]))
    model.load_state_dict(ckpt["model"])
    tasks = make_prompts(build_tasks(seed))
    result = {
        "schema": 1, "seed": seed, "model_step": ckpt["step"],
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest(),
        "score": score(model, tokenizer, tasks),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=941027)
    args = parser.parse_args()
    evaluate(args.checkpoint_dir, args.out, args.seed)


if __name__ == "__main__":
    main()
