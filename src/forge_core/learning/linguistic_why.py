"""Gold-verified linguistic *why* probes, isolated from personal why/memory.

This generates ORIGINAL controlled English minimal pairs. It is a diagnostic
curriculum/evaluation fixture, NOT an empirical language-understanding score,
and no model output is ever automatically accepted as a gold explanation.

Do not train on the evaluation split. A separate external evaluation corpus is
required before claiming general linguistic competence.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

# No shared domain vocabulary between the split-specific sentence actors.
_LEXICONS = {
    "train": ("fox", "rabbit", "otter", "deer", "bear", "owl"),
    "eval": ("pilot", "gardener", "sailor", "teacher", "chef", "artist"),
}
# Original rule templates; these are NOT imported from a third-party corpus.
_VERBS = (
    ("follow", "follows", "followed"),
    ("chase", "chases", "chased"),
    ("help", "helps", "helped"),
    ("watch", "watches", "watched"),
    ("find", "finds", "found"),
)

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
    source: str = "Forge Core original controlled minimal-pair fixtures"

    def public_question(self) -> dict[str, Any]:
        """Hide gold answers/rule when an evaluator queries a candidate model."""
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
    for index, (root, present, past) in enumerate(_VERBS):
        actor = words[index % len(words)]
        patient = words[(index + 2) % len(words)]
        # Both grammatical; subject-object reversal flips the event roles.
        a = f"The {actor} {past} the {patient}."
        b = f"The {patient} {past} the {actor}."
        probes.append(LinguisticProbe(
            id=f"{split}:roles:{index}", split=split, skill="semantic_roles",
            sentence_a=a, sentence_b=b,
            question="In each sentence, who performed the action and who received it?",
            gold={
                "a": {"agent": actor, "patient": patient},
                "b": {"agent": patient, "patient": actor},
                "both_grammatical": True,
                "roles_reversed": True,
            },
            verified_rule="With an active transitive clause in this controlled template, the preverbal noun phrase is the agent and the postverbal noun phrase is the patient.",
            actors=(actor, patient),
        ))
        # Grammaticality is not the same property as semantic-role reversal.
        a = f"The {actor} {present} the {patient}."
        b = f"The {actor} {root} the {patient}."
        probes.append(LinguisticProbe(
            id=f"{split}:agreement:{index}", split=split,
            skill="subject_verb_agreement", sentence_a=a, sentence_b=b,
            question="Which sentence follows standard English subject-verb agreement?",
            gold={"grammatical": "a", "ungrammatical": "b",
                  "subject": actor, "verb_form": present},
            verified_rule="A third-person singular subject requires the third-person singular present indicative form in this controlled template.",
            actors=(actor, patient),
        ))
        # The negated variant stays grammatical but denies the event.
        a = f"The {actor} {past} the {patient}."
        b = f"The {actor} did not {root} the {patient}."
        probes.append(LinguisticProbe(
            id=f"{split}:negation:{index}", split=split, skill="negation",
            sentence_a=a, sentence_b=b,
            question="Which sentence affirms the event and which denies it?",
            gold={"a": {"event_polarity": "affirmed"}, "b": {"event_polarity": "negated"},
                  "both_grammatical": True},
            verified_rule="In this template 'did not' negates the past event; the lexical verb uses its base form.",
            actors=(actor, patient),
        ))
    return probes


def validate_splits(train: list[LinguisticProbe], evaluation: list[LinguisticProbe]) -> None:
    if not train or not evaluation:
        raise ValueError("Both splits must be populated")
    a, b = {p.id for p in train}, {p.id for p in evaluation}
    if len(a) != len(train) or len(b) != len(evaluation) or a & b:
        raise ValueError("Duplicate or overlapping probe IDs")
    if {w for p in train for w in p.actors} & {w for p in evaluation for w in p.actors}:
        raise ValueError("Evaluation vocabulary overlaps training vocabulary")
    if any(not p.verified_rule or not p.gold for p in train + evaluation):
        raise ValueError("Unverified or missing gold answer")


def write_probes(output: Path) -> dict:
    train, evaluation = build_probes("train"), build_probes("eval")
    validate_splits(train, evaluation)
    output.mkdir(parents=True, exist_ok=True)
    # Evaluator gold MUST NOT be supplied to the model as an input.
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
        "schema": 1, "origin": "Forge Core original controlled language probes",
        "license": "project-authored original fixtures; separate from third-party training data",
        "train_count": len(train), "eval_count": len(evaluation),
        "skills": sorted({p.skill for p in train}),
        "hashes": sha,
        "limitations": [
            "Very small deterministic templates, not natural-language capability evidence",
            "Only agents/patients in active clauses are tested; grammar differs by dialect/context",
            "Evaluation nouns disjoint; verb patterns and structures remain shared",
            "Never use eval-gold.jsonl in training",
            "Do not use model-generated explanations as self-verified labels",
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
