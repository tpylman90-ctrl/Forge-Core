"""Independent, NO-TRAINING role-binding diagnostic for Forge's frozen pointer.

Tests whether copying an unfamiliar name is tied to its semantic role,
not its absolute position or a familiar question template. New public
synthetic test only; no private user data, no checkpoint promotion.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re

import torch

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.preschool_qa import verify_source
from forge_core.neural.span_pointer_lab import WordPointer, features_for

# Deliberately independent from existing pointer/development/exam vocabularies.
PEOPLE = ("Nola", "Quinn", "Elio", "Vera", "Basil", "Cassia", "Leila",
          "Oren", "Priya", "Soren", "Felix", "Uma", "Celia", "Nico",
          "Petra", "Imani", "Gideon", "Lyra")
OBJECTS = ("magnet", "notebook", "puzzle", "telescope", "suitcase",
           "pillow", "ticket", "tablet", "jar", "necklace")
TEMPLATES = ("handed", "received", "passive", "sent")
ROLES = ("giver", "recipient")
SEEDS = (6101, 6102, 6103)

# These four parsers reconstruct gold independently of the case generator.
GRAMMARS = {
    "handed": re.compile(r"^([A-Za-z]+) handed the ([a-z]+) to ([A-Za-z]+)\.$"),
    "received": re.compile(r"^([A-Za-z]+) received the ([a-z]+) from ([A-Za-z]+)\.$"),
    "passive": re.compile(r"^The ([a-z]+) was given to ([A-Za-z]+) by ([A-Za-z]+)\.$"),
    "sent": re.compile(r"^([A-Za-z]+) sent the ([a-z]+) to ([A-Za-z]+)\.$"),
}


@dataclass(frozen=True)
class RoleCase:
    id: str
    skill: str
    prompt: str
    answer: str
    template: str
    seed: int
    trial: int
    order: int
    giver: str
    recipient: str


def story_for(template: str, giver: str, recipient: str, obj: str) -> str:
    return {
        "handed": f"{giver} handed the {obj} to {recipient}.",
        "received": f"{recipient} received the {obj} from {giver}.",
        "passive": f"The {obj} was given to {recipient} by {giver}.",
        "sent": f"{giver} sent the {obj} to {recipient}.",
    }[template]


def parse_roles(template: str, story: str) -> tuple[str, str, str]:
    match = GRAMMARS[template].fullmatch(story)
    if match is None:
        raise ValueError("Unexpected independent role grammar")
    a, obj, b = match.groups()
    if template == "received":
        return b, a, obj
    if template == "passive":
        # The passive parser returns (object, recipient, giver), unlike
        # active forms. Parse by named semantic slots, not tuple position.
        object_name, receiver_name, sender_name = match.groups()
        return sender_name, receiver_name, object_name
    return a, b, obj


def make_cases(seed: int, per_template: int = 8) -> list[RoleCase]:
    if seed not in SEEDS or not 2 <= per_template <= 24:
        raise ValueError("Outside preregistered role-diagnostic bounds")
    rng = random.Random(seed)
    result: list[RoleCase] = []
    for template in TEMPLATES:
        for trial in range(per_template):
            first, second = rng.sample(PEOPLE, 2)
            obj = rng.choice(OBJECTS)
            for order in (0, 1):
                giver, recipient = (first, second) if order == 0 else (second, first)
                story = story_for(template, giver, recipient, obj)
                for role in ROLES:
                    question = ("Who gave the " if role == "giver"
                                else "Who received the ") + obj + "?"
                    prompt = (f"Story: In class session {seed}, {story}\n"
                              f"Question: {question}\nAnswer:")
                    answer = giver if role == "giver" else recipient
                    result.append(RoleCase(
                        id=f"role-v1:{seed}:{template}:{trial}:{order}:{role}",
                        skill=role, prompt=prompt, answer=answer,
                        template=template, seed=seed, trial=trial,
                        order=order, giver=giver, recipient=recipient,
                    ))
    verify_cases(result)
    return result


def verify_cases(cases: list[RoleCase]) -> None:
    if not cases or len({row.id for row in cases}) != len(cases):
        raise ValueError("Missing or duplicate role-diagnostic identifiers")
    if len({row.prompt for row in cases}) != len(cases):
        raise ValueError("Duplicate diagnostic prompts")
    for row in cases:
        if row.template not in TEMPLATES or row.skill not in ROLES:
            raise ValueError("Invalid role or template")
        prefix = f"Story: In class session {row.seed}, "
        if not row.prompt.startswith(prefix) or "\nQuestion: " not in row.prompt:
            raise ValueError("Unknown source structure")
        story, suffix = row.prompt[len(prefix):].split("\nQuestion: ", 1)
        actual_giver, actual_recipient, obj = parse_roles(row.template, story)
        question = ("Who gave the " if row.skill == "giver"
                    else "Who received the ") + obj + "?\nAnswer:"
        if suffix != question:
            raise ValueError("Role query is inconsistent with the source")
        expected = actual_giver if row.skill == "giver" else actual_recipient
        if (expected != row.answer or actual_giver == actual_recipient
                or row.giver != actual_giver or row.recipient != actual_recipient):
            raise ValueError("Role answer or event facts do not match independently parsed truth")


def check_novel_lexicon() -> None:
    from forge_core.neural.preschool_copy import (
        TRAIN_PEOPLE, EVAL_PEOPLE, TRAIN_WORDS, EVAL_WORDS)
    from forge_core.neural.pointer_transfer import EXAM_PEOPLE, EXAM_WORDS
    old_people = set(TRAIN_PEOPLE) | set(EVAL_PEOPLE) | set(EXAM_PEOPLE)
    old_objects = set(TRAIN_WORDS) | set(EVAL_WORDS) | set(EXAM_WORDS)
    if set(PEOPLE) & old_people or set(OBJECTS) & old_objects:
        raise ValueError("New diagnostic vocabulary overlaps previous pointer corpora")


def load_frozen_models(source: Path, parent: Path, pointer: Path):
    """Reject changed model lineage and restore only verified frozen heads."""
    original, manifest, codec = verify_source(source)
    original_sha = hashlib.sha256((source / "checkpoint.pt").read_bytes()).hexdigest()
    parent_path = parent / "verified-research.pt"
    parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    research = torch.load(parent_path, map_location="cpu", weights_only=True)
    if (research.get("schema") != 4
            or research.get("arm") != "verified"
            or not research.get("experimental_only")
            or research.get("source_checkpoint_sha256") != original_sha
            or research.get("configuration") != original["configuration"]):
        raise ValueError("Invalid original-to-parent lineage")
    backbone = ByteTransformer(ModelConfig(**original["configuration"]))
    backbone.load_state_dict(research["model"], strict=True)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad_(False)
    heads = {}
    for arm in ("verified", "wrong-source"):
        data = torch.load(pointer / f"{arm}-pointer.pt",
                          map_location="cpu", weights_only=True)
        if (data.get("schema") != 1 or not data.get("experimental_only")
                or data.get("arm") != arm
                or data.get("source_checkpoint_sha256") != original_sha
                or data.get("parent_sha256") != parent_sha
                or data.get("width") != backbone.cfg.n_embd):
            raise ValueError("Invalid pointer head lineage")
        head = WordPointer(backbone.cfg.n_embd)
        head.load_state_dict(data["model"], strict=True)
        head.eval()
        heads[arm] = head
    return backbone, codec, heads, {
        "original_sha256": original_sha,
        "parent_sha256": parent_sha,
        "train_sha256": manifest["train_sha256"],
    }


def summarize(cases: list[RoleCase], predicted: dict[str, str]) -> dict:
    if len(predicted) != len(cases) or set(predicted) != {c.id for c in cases}:
        raise ValueError("Each diagnostic prompt needs exactly one prediction")
    by_skill = defaultdict(lambda: [0, 0])
    by_template = defaultdict(lambda: [0, 0])
    by_seed = defaultdict(lambda: [0, 0])
    name_choice = 0
    correct = 0
    for case in cases:
        guess = predicted[case.id]
        hit = int(guess.casefold() == case.answer.casefold())
        correct += hit
        name_choice += int(guess in (case.giver, case.recipient))
        for d, key in ((by_skill, case.skill),
                       (by_template, case.template), (by_seed, str(case.seed))):
            d[key][0] += hit
            d[key][1] += 1

    lookup = {(c.seed, c.template, c.trial, c.order, c.skill): c
              for c in cases}
    role_pairs = 0
    both_role_correct = 0
    role_name_discrimination = 0
    counterfactual_pairs = 0
    both_swap_correct = 0
    swap_changes = 0
    for seed in sorted({c.seed for c in cases}):
        for template in TEMPLATES:
            trials = sorted({c.trial for c in cases
                             if c.seed == seed and c.template == template})
            for trial in trials:
                for order in (0, 1):
                    giver = lookup[(seed, template, trial, order, "giver")]
                    recipient = lookup[(seed, template, trial, order, "recipient")]
                    gp, rp = predicted[giver.id], predicted[recipient.id]
                    role_pairs += 1
                    both_role_correct += int(gp == giver.answer and rp == recipient.answer)
                    role_name_discrimination += int(
                        gp != rp and gp in (giver.giver, giver.recipient)
                        and rp in (recipient.giver, recipient.recipient))
                for role in ROLES:
                    before = lookup[(seed, template, trial, 0, role)]
                    after = lookup[(seed, template, trial, 1, role)]
                    a, b = predicted[before.id], predicted[after.id]
                    counterfactual_pairs += 1
                    both_swap_correct += int(a == before.answer and b == after.answer)
                    swap_changes += int(a != b)

    def scores(d):
        return {k: {"correct": v[0], "total": v[1],
                    "accuracy": round(v[0] / v[1], 4)}
                for k, v in sorted(d.items())}
    return {
        "correct": correct, "total": len(cases),
        "accuracy": round(correct / len(cases), 4),
        "selected_one_of_two_people": name_choice,
        "by_skill": scores(by_skill), "by_template": scores(by_template),
        "by_seed": scores(by_seed),
        "role_question_pairs": {
            "total": role_pairs, "both_correct": both_role_correct,
            "selected_different_people": role_name_discrimination,
        },
        "entity_swap_pairs": {
            "total": counterfactual_pairs, "both_correct": both_swap_correct,
            "changed_predicted_word": swap_changes,
        },
    }


def diagnose(*, source: Path, parent: Path, pointer: Path, out: Path) -> dict:
    torch.set_num_threads(min(2, torch.get_num_threads()))
    check_novel_lexicon()
    cases = [case for seed in SEEDS for case in make_cases(seed)]
    verify_cases(cases)
    backbone, codec, heads, provenance = load_frozen_models(source, parent, pointer)
    predicted = {arm: {} for arm in heads}
    sample = []
    with torch.no_grad():
        for case in cases:
            item = features_for(backbone, codec, case)
            for arm, head in heads.items():
                scores = head(item.query, item.words)
                predicted[arm][case.id] = item.candidates[int(scores.argmax())]
            if len(sample) < 18:
                sample.append({
                    "id": case.id, "template": case.template,
                    "role": case.skill, "expected": case.answer,
                    "verified_prediction": predicted["verified"][case.id],
                    "control_prediction": predicted["wrong-source"][case.id],
                })
    report = {
        "schema": 1, "type": "diagnostic-role-binding-counterfactual",
        "research_only": True, "no_training": True, "promoted": False,
        "source_lineage": provenance,
        "seeds": list(SEEDS), "cases": len(cases),
        "frozen_backbone": all(not p.requires_grad for p in backbone.parameters()),
        "verified_pointer": summarize(cases, predicted["verified"]),
        "wrong_source_control": summarize(cases, predicted["wrong-source"]),
        "public_synthetic_examples": sample,
        "limits": [
            "Four hand-written grammar templates, not independent natural-language evidence",
            "No new weights and no ability claim, even if rule-controlled questions succeed",
            "Two-choice person baselines are not equivalent to all-word pointer selection",
            "Only diagnostic syntax families; do not tune repeatedly on these cases",
            "Model may copy an unfamiliar name without understanding a real event",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "cases": len(cases),
        "verified": report["verified_pointer"],
        "control": report["wrong_source_control"],
        "no_training": True,
    }, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--pointer", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/role-binding-diagnostic.json"))
    args = parser.parse_args()
    diagnose(source=args.source, parent=args.parent, pointer=args.pointer, out=args.out)


if __name__ == "__main__":
    main()
