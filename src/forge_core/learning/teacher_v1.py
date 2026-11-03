"""Forge Teacher v1: a *verified*, adaptive curriculum for scratch-trained Forge.

A pretrained teacher may recommend which skill to teach (via a separately
installed local GGUF). It NEVER supplies trusted answers, changes weights,
writes personal memories or acts independently. Gold examples are generated
by Forge Core's original, deterministic, programmatically checked tasks.
No remote APIs and no personal observations are read by this module.

Progress input MUST be aggregate DEVELOPMENT scores, never final exam labels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random

from forge_core.neural.preschool_copy import (
    SKILLS, TRAIN_PEOPLE, EVAL_PEOPLE, TRAIN_WORDS, EVAL_WORDS, tasks,
)

RULES = {
    "copy": "Copy the specified source word exactly. Other words are distractors.",
    "select": "For a one-indexed list, return the item at the requested position.",
    "recipient": "The receiver is the person who gets the transferred object.",
    "giver": "The giver is the person who transfers the object to the receiver.",
    "color": "Use the color associated with the person asked about, not the other person's color.",
    "quantity": "Use the number associated with the person asked about, not the other person's count.",
}
SKILL_SET = frozenset(SKILLS)


@dataclass(frozen=True)
class Lesson:
    id: str
    skill: str
    prompt: str
    answer: str
    verified_rule: str
    source: str = "Forge Core original rule-labeled nonpersonal curriculum"


def _json(data: object) -> bytes:
    return (json.dumps(data, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_development_scores(scores: dict | None) -> dict[str, dict[str, int]]:
    """Accept small aggregate counts only; no personal observations or eval gold."""
    if scores is None:
        return {skill: {"correct": 0, "total": 0} for skill in SKILLS}
    if not isinstance(scores, dict) or set(scores) != SKILL_SET:
        raise ValueError("Development scores must contain exactly six known skills")
    checked = {}
    for skill in SKILLS:
        value = scores[skill]
        if (not isinstance(value, dict) or
            set(value) != {"correct", "total"} or
            type(value["correct"]) is not int or type(value["total"]) is not int):
            raise ValueError("Only aggregate integer correct/total counts allowed")
        correct, total = value["correct"], value["total"]
        if not 0 <= correct <= total <= 10000:
            raise ValueError("Invalid development skill score")
        checked[skill] = {"correct": correct, "total": total}
    return checked


def validate_teacher_proposal(proposal: dict | None) -> dict | None:
    """Teacher chooses a focus, not labels, datasets, weights, or private facts."""
    if proposal is None:
        return None
    if (not isinstance(proposal, dict) or set(proposal) != {"focus_skill", "reason"}
        or proposal["focus_skill"] not in SKILL_SET
        or not isinstance(proposal["reason"], str)
        or not 1 <= len(proposal["reason"]) <= 180
        or any(ch in proposal["reason"] for ch in "\n\r\x00")):
        raise ValueError("Untrusted teacher proposal outside approved schema")
    return {"focus_skill": proposal["focus_skill"],
            "reason": proposal["reason"]}


def plan_quotas(count: int, scores: dict[str,dict[str,int]],
                proposal: dict | None = None) -> dict[str,int]:
    if not 60 <= count <= 1200 or count % len(SKILLS):
        raise ValueError("Training count must be 60..1200 and multiple of six")
    # Shrinkage avoids overreacting to 1/1 and 0/1 development results.
    # Min weight prevents mastered skills from being discarded entirely.
    weights = {}
    for skill in SKILLS:
        correct, total = scores[skill]["correct"], scores[skill]["total"]
        adjusted = (correct + 2) / (total + 4)
        weights[skill] = max(0.35, 1.15 - adjusted)
    if proposal:
        weights[proposal["focus_skill"]] *= 1.3  # bounded advisory impact
    quota = {skill: 1 for skill in SKILLS}
    for _ in range(count - len(SKILLS)):
        next_skill = max(SKILLS, key=lambda k: (
            weights[k] / (quota[k] + 1), -SKILLS.index(k)))
        quota[next_skill] += 1
    return quota


def make_plan(*, seed: int = 1701, train_count: int = 240,
              dev_count: int = 120,
              progress: dict | None = None,
              teacher_proposal: dict | None = None) -> tuple[dict,list[Lesson],list[Lesson]]:
    if not 1 <= seed <= 1_000_000:
        raise ValueError("Seed must be bounded and reproducible")
    if not 60 <= dev_count <= 600 or dev_count % len(SKILLS):
        raise ValueError("Development set size outside allowed range")
    if set(TRAIN_PEOPLE) & set(EVAL_PEOPLE) or set(TRAIN_WORDS) & set(EVAL_WORDS):
        raise ValueError("Training and development lexicons overlap")
    scores = validate_development_scores(progress)
    proposal = validate_teacher_proposal(teacher_proposal)
    quotas = plan_quotas(train_count, scores, proposal)
    pool_count = max(600, train_count * 4)
    pool_count += (-pool_count) % len(SKILLS)
    # One deterministic pool per seed. Selection only changes which trusted
    # examples are used, never the rule-derived correct answer itself.
    pool = tasks("train", pool_count, seed)
    by_skill = {skill: [row for row in pool if row.skill == skill]
                for skill in SKILLS}
    rng = random.Random(seed + 711)
    selected = []
    for skill in SKILLS:
        if quotas[skill] > len(by_skill[skill]):
            raise ValueError("Not enough verified examples for lesson quota")
        selected.extend(rng.sample(by_skill[skill], quotas[skill]))
    rng.shuffle(selected)
    dev = tasks("challenge", dev_count, seed + 1001)
    if {row.prompt for row in selected} & {row.prompt for row in dev}:
        raise ValueError("Training and development prompt overlap")
    train_lessons = [Lesson(row.id, row.skill, row.prompt, row.answer,
                            RULES[row.skill]) for row in selected]
    dev_lessons = [Lesson(row.id, row.skill, row.prompt, row.answer,
                          RULES[row.skill]) for row in dev]
    plan = {
        "schema": 1, "mode": "verified-teacher-lesson-selection",
        "seed": seed, "train_count": train_count, "development_count": dev_count,
        "source_pool_count": pool_count,
        "progress_source": "aggregate_development_scores_only",
        "dev_scores": scores, "teacher_advice": proposal,
        "quotas": quotas, "train_ids": [row.id for row in selected],
        "source": "Forge Core original synthetic deterministic rule-labeled instructions",
        "teacher_can_change_gold_answers": False,
        "uses_private_memory": False, "uses_pretrained_student_weights": False,
        "ready_for_promotion": False,
        "limitations": [
            "This is adaptive *selection*, not proof a local pretrained AI ran",
            "Rule labels are verified only within the defined synthetic templates",
            "Development data may guide lessons; FINAL independent exams must stay sealed",
            "Original scratch-trained neural weights are unchanged by lesson generation",
        ],
    }
    return plan, train_lessons, dev_lessons


def write_bundle(folder: Path, *, seed: int = 1701, train_count: int = 240,
                 dev_count: int = 120, progress: dict | None = None,
                 teacher_proposal: dict | None = None) -> dict:
    plan, train, dev = make_plan(
        seed=seed, train_count=train_count, dev_count=dev_count,
        progress=progress, teacher_proposal=teacher_proposal)
    folder.mkdir(parents=True, exist_ok=True)
    files = {
        "lesson-plan.json": _json(plan),
        "train.jsonl": b"".join(_json(asdict(row)) for row in train),
        # Development questions deliberately omit gold labels.
        "development-questions.jsonl": b"".join(_json({
            "id":row.id, "skill":row.skill, "prompt":row.prompt
        }) for row in dev),
        "development-gold.jsonl": b"".join(_json({
            "id":row.id, "answer":row.answer, "verified_rule":row.verified_rule
        }) for row in dev),
    }
    for filename, data in files.items():
        (folder / filename).write_bytes(data)
    manifest = {
        "schema": 1, "source": plan["source"], "teacher_weights_used": False,
        "verified_labels_from_original_rules": True,
        "train_count": len(train), "development_count": len(dev),
        "files_sha256": {name: _digest(data) for name,data in files.items()},
        "no_personal_data": True, "research_only": True,
    }
    (folder / "manifest.json").write_bytes(_json(manifest))
    return manifest


def audit_bundle(folder: Path) -> tuple[list[Lesson],list[Lesson]]:
    manifest = json.loads((folder/"manifest.json").read_text("utf-8"))
    plan = json.loads((folder/"lesson-plan.json").read_text("utf-8"))
    if manifest.get("schema") != 1 or not manifest.get("verified_labels_from_original_rules"):
        raise ValueError("Unapproved teacher bundle")
    if set(manifest.get("files_sha256", {})) != {
        "lesson-plan.json","train.jsonl","development-questions.jsonl",
        "development-gold.jsonl"}:
        raise ValueError("Missing or unexpected lesson files")
    for filename,digest in manifest["files_sha256"].items():
        if _digest((folder/filename).read_bytes()) != digest:
            raise ValueError("Teacher bundle content hash mismatch")
    expected, train, dev = make_plan(
        seed=plan["seed"], train_count=plan["train_count"],
        dev_count=plan["development_count"], progress=plan["dev_scores"],
        teacher_proposal=plan["teacher_advice"])
    if expected != plan:
        raise ValueError("Teacher plan differs from trusted curriculum algorithm")
    if (folder/"train.jsonl").read_bytes() != b"".join(_json(asdict(x)) for x in train):
        raise ValueError("Teacher training labels differ from verified rules")
    if (folder/"development-questions.jsonl").read_bytes() != b"".join(
        _json({"id":x.id,"skill":x.skill,"prompt":x.prompt}) for x in dev):
        raise ValueError("Development questions were changed")
    if (folder/"development-gold.jsonl").read_bytes() != b"".join(
        _json({"id":x.id,"answer":x.answer,"verified_rule":x.verified_rule}) for x in dev):
        raise ValueError("Development gold labels were changed")
    return train, dev


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("artifacts/teacher-v1"))
    p.add_argument("--seed",type=int,default=1701)
    p.add_argument("--train-count",type=int,default=240)
    p.add_argument("--dev-count",type=int,default=120)
    p.add_argument("--progress-json",type=Path)
    p.add_argument("--teacher-advice-json",type=Path)
    args=p.parse_args()
    progress=json.loads(args.progress_json.read_text()) if args.progress_json else None
    advice=json.loads(args.teacher_advice_json.read_text()) if args.teacher_advice_json else None
    manifest=write_bundle(args.out,seed=args.seed,train_count=args.train_count,
                          dev_count=args.dev_count,progress=progress,
                          teacher_proposal=advice)
    audit_bundle(args.out)
    print(json.dumps(manifest,indent=2))


if __name__ == "__main__":
    main()
