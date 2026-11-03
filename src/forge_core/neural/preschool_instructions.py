"""Independent one-step instruction audit for original Forge Seed v0.7.

Five original synthetic command categories, ten examples each. These are
never used to teach the model. Exact first-answer match is intentionally
strict. This is a diagnostic, not an educational or AI capability certificate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random

import torch

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.preschool_direct import DirectExample, evaluate_direct
from forge_core.neural.preschool_qa import verify_source

AUDIT_SKILLS = ("repeat", "select", "color", "quantity", "recipient")
PEOPLE = ("Jace", "Tess", "Dina", "Kian", "Omar",
          "Mara", "Ida", "Eli", "Kai", "Bea")
WORDS = ("garden", "river", "lamp", "tree", "stone",
         "apple", "rain", "bird", "moon", "chair")
COLOR_WORDS = ("red", "blue", "yellow", "green", "purple",
               "white", "orange", "brown", "black", "pink")
QUANTITY_WORDS = ("two", "three", "four", "five", "six",
                  "seven", "eight", "nine", "ten", "eleven")


def instruction_examples(seed: int = 233) -> list[DirectExample]:
    rng = random.Random(seed)
    rows = []
    for skill in AUDIT_SKILLS:
        for i in range(10):
            name, other = rng.sample(PEOPLE, 2)
            word, another, third = rng.sample(WORDS, 3)
            if skill == "repeat":
                prompt = (
                    f"Instruction: Write only the word {word}.\nAnswer:"
                    if i % 2 == 0 else
                    f"Instruction: Repeat this word without extra words: {word}.\nAnswer:"
                )
                answer = word
            elif skill == "select":
                choices = (word, another, third)
                index = i % 3
                ordinal = ("first", "second", "third")[index]
                prompt = (f"Instruction: Here are three words: "
                          f"{choices[0]}, {choices[1]}, {choices[2]}. "
                          f"Give the {ordinal} word only.\nAnswer:")
                answer = choices[index]
            elif skill == "color":
                c1, c2 = rng.sample(COLOR_WORDS, 2)
                target_second = i % 2 == 1
                prompt = (f"Instruction: In this sentence, {name} has a "
                          f"{c1} ribbon and {other} has a {c2} flag. "
                          f"Reply with the color belonging to "
                          f"{other if target_second else name}, "
                          f"and nothing else.\nAnswer:")
                answer = c2 if target_second else c1
            elif skill == "quantity":
                c1, c2 = rng.sample(QUANTITY_WORDS, 2)
                target_second = i % 2 == 1
                prompt = (f"Instruction: {name} collected {c1} stones "
                          f"while {other} collected {c2} shells. "
                          f"Say only the number word for "
                          f"{other if target_second else name}.\nAnswer:")
                answer = c2 if target_second else c1
            else:
                if i % 2 == 0:
                    story = f"{name} handed the book to {other}."
                else:
                    story = f"{other} received the book from {name}."
                prompt = (
                    f"Instruction: Read this sentence: {story} "
                    f"Write only the name of the recipient.\nAnswer:"
                )
                answer = other
            rows.append(DirectExample(
                f"instruction:{skill}:{i:02d}", skill, prompt, answer))
    if len({row.prompt for row in rows}) != 50:
        raise ValueError("Instruction prompts not unique")
    return rows


def examine(base_dir: Path, result_dir: Path, out: Path, seed: int = 233) -> dict:
    torch.set_num_threads(min(2, torch.get_num_threads()))
    base_ckpt, manifest, tokenizer = verify_source(base_dir)
    config = ModelConfig(**base_ckpt["configuration"])
    base_hash = hashlib.sha256((base_dir / "checkpoint.pt").read_bytes()).hexdigest()
    tasks = instruction_examples(seed)
    challenge_hash = hashlib.sha256(
        "\n".join(x.prompt + " " + x.answer for x in tasks).encode()
    ).hexdigest()

    def from_weights(state):
        model = ByteTransformer(config)
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    source = evaluate_direct(from_weights(base_ckpt["model"]), tokenizer, tasks)
    arms = {}
    for arm in ("verified", "shuffled"):
        path = result_dir / f"{arm}-research.pt"
        if not path.is_file():
            raise ValueError("Candidate artifact is missing " + arm)
        weights = torch.load(path, map_location="cpu", weights_only=True)
        if weights.get("schema") != 4 or not weights.get("experimental_only"):
            raise ValueError("Unapproved research checkpoint schema")
        if weights.get("source_checkpoint_sha256") != base_hash:
            raise ValueError("Candidate originated from wrong source checkpoint")
        if weights.get("configuration") != base_ckpt["configuration"]:
            raise ValueError("Candidate model config differs from original")
        if weights.get("arm") != arm:
            raise ValueError("Research arm labeling incorrect")
        arms[arm] = evaluate_direct(
            from_weights(weights["model"]), tokenizer, tasks)
    report = {
        "schema": 1, "status": "independent_single_step_instructions_not_certified",
        "source_checkpoint_sha256": base_hash,
        "source_train_sha256": manifest["train_sha256"],
        "challenge_hash": challenge_hash,
        "examples": len(tasks), "skills": list(AUDIT_SKILLS),
        "original": source, "verified_teaching": arms["verified"],
        "shuffled_control": arms["shuffled"],
        "no_candidate_promoted": True,
        "limitations": [
            "Only 50 original fixed synthetic instructions; not educational certification",
            "Not used in the adaptation training or optimization",
            "Exact-match short responses may penalize reasonable synonyms",
            "A single training seed cannot establish reproducible instruction following",
            "Answer words, not open-ended free-form conversational quality",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "original_correct": source["correct"],
        "verified_correct": arms["verified"]["correct"],
        "shuffled_correct": arms["shuffled"]["correct"],
        "total": len(tasks),
        "by_skill": arms["verified"]["by_skill"],
    }, indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--candidate-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("artifacts/instruction-audit.json"))
    a = p.parse_args()
    examine(a.checkpoint_dir, a.candidate_dir, a.out)


if __name__ == "__main__":
    main()
