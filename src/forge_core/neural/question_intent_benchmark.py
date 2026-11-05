"""Independent question-only role-intent benchmark, version 1.

Hand-authored question families; balanced labels; no stories or pretrained teacher.
The final split uses disjoint wording families and nouns and is never used by a
training routine in this module. This is an internal research benchmark, not a
standardized language-comprehension test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from collections import Counter

LABELS = ("giver", "recipient", "other")

TRAIN_ITEMS = ("lantern", "map", "cup", "scarf", "drum", "shell", "basket", "key")
DEV_ITEMS = ("marble", "helmet", "brush", "notebook")
FINAL_ITEMS = ("bell", "compass", "ribbon", "badge", "glove")

# All wording families below were authored for this benchmark. Labels describe
# the transfer role the question explicitly asks for. "other" is a negative
# class: an ordinary action or location question with no transfer role.
TRAIN_FAMILIES = {
    "giver": (
        "Who gave the {item} to another person?",
        "Who handed over the {item}?",
        "Which person passed the {item} along?",
        "Who sent the {item} to somebody else?",
        "Who supplied the {item}?",
        "Who let another person have the {item}?",
        "Which person parted with the {item}?",
        "Who donated the {item} to someone?",
    ),
    "recipient": (
        "Who received the {item} from somebody else?",
        "Which person was given the {item}?",
        "Who ended up with the {item}?",
        "Who got the {item} from another person?",
        "Which person was handed the {item}?",
        "Who accepted the {item} from someone?",
        "To whom was the {item} given?",
        "Who took possession of the {item}?",
    ),
    "other": (
        "Who found the {item}?",
        "Which person opened the {item}?",
        "Who carried the {item}?",
        "Who looked at the {item}?",
        "Who repaired the {item}?",
        "Which person stood near the {item}?",
        "Who described the {item}?",
        "Who drew the {item}?",
    ),
}

DEV_FAMILIES = {
    "giver": (
        "Who passed the {item} from one person to another?",
        "Which person gave away the {item}?",
        "Who was the one who transferred the {item} out?",
        "Who parted from the {item} by giving it to someone?",
        "Which person sent the {item} on to somebody?",
    ),
    "recipient": (
        "Who had the {item} passed to them?",
        "Which person took the {item} from its giver?",
        "Who was the destination for the {item}?",
        "Who got the {item} when it changed hands?",
        "To which person did someone give the {item}?",
    ),
    "other": (
        "Who displayed the {item}?",
        "Which person cleaned the {item}?",
        "Who placed the {item} on a shelf?",
        "Who was looking for the {item}?",
        "Which person painted the {item}?",
    ),
}

FINAL_FAMILIES = {
    "giver": (
        "Who was the source of the {item} in the exchange?",
        "Which person transferred the {item} to someone else?",
        "From whom did another person get the {item}?",
        "Who started the transfer by giving away the {item}?",
        "Who let the {item} go to another person?",
        "Which person was responsible for giving out the {item}?",
    ),
    "recipient": (
        "Who came into possession of the {item} from another person?",
        "To whose care did the {item} go?",
        "Which person got the {item} when it changed hands?",
        "Who was the endpoint of the {item}'s transfer?",
        "Who did someone hand the {item} to?",
        "Which person took the {item} from its giver?",
    ),
    "other": (
        "Who put the {item} on the table?",
        "Which person carried the {item} upstairs?",
        "Who noticed the {item} first?",
        "Who described the {item} to a friend?",
        "Who stood beside the {item}?",
        "Which person used the {item} to open a door?",
    ),
}


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.casefold()))


def expand(split: str, families: dict[str, tuple[str, ...]],
           items: tuple[str, ...]) -> list[dict[str, str]]:
    rows = []
    for label in LABELS:
        for family_index, template in enumerate(families[label]):
            for item in items:
                question = template.format(item=item)
                rows.append({
                    "id": f"{split}:{label}:{family_index}:{item}",
                    "split": split,
                    "family_id": f"{split}:{label}:{family_index}",
                    "question": question,
                    "label": label,
                })
    return rows


def build() -> dict[str, list[dict[str, str]]]:
    splits = {
        "train": expand("train", TRAIN_FAMILIES, TRAIN_ITEMS),
        "development": expand("development", DEV_FAMILIES, DEV_ITEMS),
        "final": expand("final", FINAL_FAMILIES, FINAL_ITEMS),
    }
    validate(splits)
    return splits


def validate(splits: dict[str, list[dict[str, str]]]) -> None:
    if tuple(splits) != ("train", "development", "final"):
        raise ValueError("Unexpected split ordering")
    expected = {"train": 192, "development": 60, "final": 90}
    all_questions: dict[str, str] = {}
    seen_families: dict[str, set[str]] = {}
    for split_name, rows in splits.items():
        if len(rows) != expected[split_name]:
            raise ValueError(f"{split_name} has {len(rows)}, expected {expected[split_name]}")
        counts = Counter(row["label"] for row in rows)
        if counts != Counter({label: expected[split_name] // 3 for label in LABELS}):
            raise ValueError(f"{split_name} labels are not balanced: {counts}")
        families = {row["family_id"] for row in rows}
        seen_families[split_name] = families
        for row in rows:
            key = normalize(row["question"])
            if not key or key in all_questions:
                raise ValueError(f"Empty or repeated question across splits: {key!r}")
            all_questions[key] = split_name
            if row["label"] not in LABELS:
                raise ValueError(f"Unknown label {row['label']}")
    if (seen_families["train"] & seen_families["development"] or
            seen_families["train"] & seen_families["final"] or
            seen_families["development"] & seen_families["final"]):
        raise ValueError("Wording family leaked across splits")
    for a, b in (("train", "development"), ("train", "final"),
                 ("development", "final")):
        a_items = {row["question"].split()[-1].rstrip("?") for row in splits[a]}
        b_items = {row["question"].split()[-1].rstrip("?") for row in splits[b]}
        if a_items & b_items:
            raise ValueError(f"Object vocabulary leaked across {a}/{b}")


def write(out: Path) -> dict:
    splits = build()
    out.mkdir(parents=True, exist_ok=True)
    file_hashes = {}
    for split, rows in splits.items():
        path = out / f"{split}.jsonl"
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        path.write_text(payload, encoding="utf-8")
        file_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema": 1,
        "benchmark": "forge-question-intent-v1",
        "labels": list(LABELS),
        "split_counts": {key: len(value) for key, value in splits.items()},
        "label_counts": {
            key: dict(Counter(row["label"] for row in value))
            for key, value in splits.items()
        },
        "files_sha256": file_hashes,
        "limitations": [
            "Internally authored, synthetic question-only wording; not externally standardized.",
            "This isolates question intent and does not test story reading or event-role extraction.",
            "Final wording is disjoint from training and development but remains visible in public source.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write(args.out), indent=2))


if __name__ == "__main__":
    main()
