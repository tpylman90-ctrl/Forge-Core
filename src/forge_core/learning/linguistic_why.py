"""Gold-verified linguistic 'why' probes for controlled structural experiments.

All examples are original, generated from explicit rule templates. The
candidate model never supplies its own training labels. The held-out split
uses different nouns AND verbs; sentence templates still overlap and cannot
support broad claims of grammatical understanding.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

_LEXICONS = {
    "train": ("fox", "rabbit", "otter", "deer", "bear", "owl"),
    "eval": ("pilot", "gardener", "sailor", "teacher", "chef", "artist"),
}
_VERBS = {
    "train": (
        ("follow", "follows", "followed"),
        ("chase", "chases", "chased"),
        ("help", "helps", "helped"),
        ("watch", "watches", "watched"),
        ("find", "finds", "found"),
    ),
    "eval": (
        ("carry", "carries", "carried"),
        ("paint", "paints", "painted"),
        ("lift", "lifts", "lifted"),
        ("push", "pushes", "pushed"),
        ("call", "calls", "called"),
    ),
}


@dataclass(frozen=True)
class LinguisticProbe:
    id: str
    split: str
    skill: str
    sentence_a: str
    sentence_b: str
    question: str
    gold: dict[str, Any]
    verified_rule: str
    actors: tuple[str, str]
    verb_root: str
    source: str = "Forge Core original controlled minimal-pair fixtures"

    def public_question(self) -> dict[str, Any]:
        return {
            "id": self.id, "skill": self.skill,
            "sentence_a": self.sentence_a, "sentence_b": self.sentence_b,
            "question": self.question,
        }


def build_probes(split: str) -> list[LinguisticProbe]:
    if split not in _LEXICONS:
        raise ValueError("Split must be train or eval")
    words = _LEXICONS[split]
    probes: list[LinguisticProbe] = []
    for index, (root, present, past) in enumerate(_VERBS[split]):
        actor = words[index % len(words)]
        patient = words[(index + 2) % len(words)]
        # Both clauses are grammatical; their semantic roles are exchanged.
        probes.append(LinguisticProbe(
            id=f"{split}:roles:{index}", split=split, skill="semantic_roles",
            sentence_a=f"The {actor} {past} the {patient}.",
            sentence_b=f"The {patient} {past} the {actor}.",
            question="Who performed the action in each sentence?",
            gold={
                "a": {"agent": actor, "patient": patient},
                "b": {"agent": patient, "patient": actor},
                "both_grammatical": True, "roles_reversed": True,
            },
            verified_rule="In this controlled active-clause template, the subject noun phrase is the agent and the object noun phrase is the patient.",
            actors=(actor, patient), verb_root=root,
        ))
        grammatical = f"The {actor} {present} the {patient}."
        incorrect = f"The {actor} {root} the {patient}."
        a, b = (grammatical, incorrect) if index % 2 == 0 else (incorrect, grammatical)
        probes.append(LinguisticProbe(
            id=f"{split}:agreement:{index}", split=split,
            skill="subject_verb_agreement", sentence_a=a, sentence_b=b,
            question="Which sentence follows standard written English singular subject-verb agreement?",
            gold={"grammatical": "a" if index % 2 == 0 else "b",
                  "ungrammatical": "b" if index % 2 == 0 else "a",
                  "subject": actor, "verb_form": present},
            verified_rule="A third-person singular present-tense subject takes the corresponding singular present verb form in this controlled template.",
            actors=(actor, patient), verb_root=root,
        ))
        affirm = f"The {actor} {past} the {patient}."
        deny = f"The {actor} did not {root} the {patient}."
        a, b = (affirm, deny) if index % 2 == 0 else (deny, affirm)
        probes.append(LinguisticProbe(
            id=f"{split}:negation:{index}", split=split, skill="negation",
            sentence_a=a, sentence_b=b,
            question="Which sentence denies that the event happened?",
            gold={"a": {"event_polarity": "affirmed" if index % 2 == 0 else "negated"},
                  "b": {"event_polarity": "negated" if index % 2 == 0 else "affirmed"},
                  "both_grammatical": True},
            verified_rule="In this template, 'did not' negates a past event while the lexical verb takes its base form.",
            actors=(actor, patient), verb_root=root,
        ))
    return probes


def validate_splits(train: list[LinguisticProbe], evaluation: list[LinguisticProbe]) -> None:
    if not train or not evaluation:
        raise ValueError("Both splits must be populated")
    a, b = {p.id for p in train}, {p.id for p in evaluation}
    if len(a) != len(train) or len(b) != len(evaluation) or a & b:
        raise ValueError("Duplicate or overlapping probe IDs")
    if {w for p in train for w in p.actors} & {w for p in evaluation for w in p.actors}:
        raise ValueError("Evaluation nouns overlap training nouns")
    if {p.verb_root for p in train} & {p.verb_root for p in evaluation}:
        raise ValueError("Evaluation verbs overlap training verbs")
    if any(not p.verified_rule or not p.gold for p in train + evaluation):
        raise ValueError("Unverified or missing gold answer")


def write_probes(output: Path) -> dict:
    train, evaluation = build_probes("train"), build_probes("eval")
    validate_splits(train, evaluation)
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "train.jsonl": [asdict(p) for p in train],
        "eval-questions.jsonl": [p.public_question() for p in evaluation],
        "eval-gold.jsonl": [{"id": p.id, "gold": p.gold, "verified_rule": p.verified_rule}
                            for p in evaluation],
    }
    sha = {}
    for name, rows in files.items():
        data = "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows)
        (output / name).write_text(data, encoding="utf-8")
        sha[name] = hashlib.sha256(data.encode("utf-8")).hexdigest()
    manifest = {
        "schema": 2, "origin": "Forge Core original controlled language probes",
        "license": "Project-authored original fixtures, separate from other public language data",
        "train_count": len(train), "eval_count": len(evaluation),
        "skills": sorted({p.skill for p in train}), "hashes": sha,
        "limits": [
            "Small deterministic templates; not natural-language comprehension evidence",
            "Disjoint evaluation nouns and verbs but shared grammatical templates",
            "Evaluation gold answers never supplied to the candidate during training",
            "Never train on eval-gold.jsonl or model-generated explanations",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/linguistic-probes"))
    args = parser.parse_args()
    print(json.dumps(write_probes(args.out), indent=2))


if __name__ == "__main__":
    main()
