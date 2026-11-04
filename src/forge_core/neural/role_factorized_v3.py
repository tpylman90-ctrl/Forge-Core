"""Forge Seed v3: separately learn story roles and question intent.

Original Forge Seed weights are source-audited, frozen and NEVER promoted.
We compare factorized semantic supervision, a deliberately wrong-story
control, query ablation, and the prior monolithic neural architecture.

All synthetic gold is reconstructed with independent source grammars.
No pretrained teacher, Android observation or personal memories are used.
"""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import resource
import time

import torch
from torch import nn
from torch.nn import functional as F

from forge_core.neural.role_binding_diagnostic import load_frozen_models
from forge_core.neural.role_conditioned_v1 import score_rows
from forge_core.neural.role_contrast_v2 import (
    ContrastRoleHead, EXAM_NAMES as V2_NAMES, EXAM_OBJECTS as V2_OBJECTS,
    Feature, _prior_vocab, features, group_flip_null, groups, measure,
    pair_loss, question_grams,
)
from forge_core.neural.role_curriculum_v1 import (
    RoleRow, candidate_spans, make_rows, verify_disjoint,
)

SEEDS = (8311, 8312)
EXAM_SEEDS = (12311, 12312)
EXAM_NAMES = ("Dalena", "Omaris", "Sariah", "Bento", "Calder", "Nadine",
              "Mikkel", "Iris", "Yvette", "Samir", "Eamon", "Tilda",
              "Perelda", "Marina", "Neron", "Belinda")
EXAM_OBJECTS = ("sundial", "pulley", "brooch", "rudder", "spool",
                "trinket", "carafe", "flask", "sandal", "amulet")
# Every new final exam template has an independently anchored grammar.
EXAM_SYNTAX = (
    ("courtyard", "in the courtyard, {giver} put the {obj} into {recipient}'s care.",
     r"in the courtyard, (?P<giver>[A-Z][a-z]+) put the (?P<obj>[a-z]+) into (?P<recipient>[A-Z][a-z]+)'s care\."),
    ("observatory", "at the observatory, {recipient} obtained the {obj} from {giver}.",
     r"at the observatory, (?P<recipient>[A-Z][a-z]+) obtained the (?P<obj>[a-z]+) from (?P<giver>[A-Z][a-z]+)\."),
    ("riverside", "beside the river, {giver} let {recipient} keep the {obj}.",
     r"beside the river, (?P<giver>[A-Z][a-z]+) let (?P<recipient>[A-Z][a-z]+) keep the (?P<obj>[a-z]+)\."),
)
# V3 words and sentence families do not appear in prior synthetic exams.
EXAM_QUESTIONS = {
    "giver": (
        "Which person initially possessed the {obj}?",
        "Who was the person that parted with the {obj}?",
    ),
    "recipient": (
        "Which person possessed the {obj} afterward?",
        "Who was the person that gained the {obj}?",
    ),
}
# More question phrasing for the SAME verified event facts. The augmentation
# is known in advance, and only TRAINING material uses these variants.
TRAIN_QUESTIONS = {
    "giver": (
        "Who gave the {obj} to someone else?",
        "Who started out with the {obj}?",
        "Who supplied the {obj}?",
        "Who handed over the {obj}?",
        "Which person owned the {obj} before the exchange?",
    ),
    "recipient": (
        "Who received the {obj} from someone else?",
        "Who ended up with the {obj}?",
        "Who took the {obj}?",
        "Who was given the {obj}?",
        "Which person owned the {obj} after the exchange?",
    ),
}


def audit_exam_vocab() -> None:
    old_names, old_objects = _prior_vocab()
    old_names.update(V2_NAMES)
    old_objects.update(V2_OBJECTS)
    if (set(EXAM_NAMES) & old_names or set(EXAM_OBJECTS) & old_objects
            or len(set(EXAM_NAMES)) != len(EXAM_NAMES)
            or len(set(EXAM_OBJECTS)) != len(EXAM_OBJECTS)):
        raise ValueError("V3 exam leaked names or objects from earlier datasets")


def source_facts(template: str, story: str) -> dict[str, str]:
    candidates = {name: pattern for name, _, pattern in EXAM_SYNTAX}
    if template not in candidates:
        raise ValueError("Unknown independent source grammar")
    parsed = re.fullmatch(candidates[template], story)
    if parsed is None:
        raise ValueError("Story does not match independent source grammar")
    return parsed.groupdict()


def make_new_exam() -> list[RoleRow]:
    audit_exam_vocab()
    output: list[RoleRow] = []
    seen: set[str] = set()
    for seed in EXAM_SEEDS:
        rng = random.Random(seed)
        for template, sentence, _ in EXAM_SYNTAX:
            for trial in range(12):
                for attempt in range(128):
                    a, b = rng.sample(EXAM_NAMES, 2)
                    obj = rng.choice(EXAM_OBJECTS)
                    proposed: list[RoleRow] = []
                    for swap in (0, 1):
                        giver, recipient = (a, b) if swap == 0 else (b, a)
                        story = sentence.format(
                            giver=giver, recipient=recipient, obj=obj)
                        truth = source_facts(template, story)
                        if truth != {"giver": giver, "recipient": recipient, "obj": obj}:
                            raise ValueError("Gold truth is inconsistent with grammar")
                        for role in ("giver", "recipient"):
                            question = EXAM_QUESTIONS[role][trial % 2].format(obj=obj)
                            prompt = f"Story: {story}\nQuestion: {question}\nAnswer:"
                            proposed.append(RoleRow(
                                id=f"v3:{seed}:{template}:{trial}:{swap}:{role}",
                                split="v3-exam", template=template, story=story,
                                prompt=prompt, answer=truth[role], role=role,
                                giver=truth["giver"], recipient=truth["recipient"],
                                object_name=obj, seed=seed, trial=trial, swap=swap,
                            ))
                    if len({r.prompt for r in proposed}) == 4 and not (
                            {r.prompt for r in proposed} & seen):
                        seen.update(r.prompt for r in proposed)
                        output.extend(proposed)
                        break
                else:
                    raise ValueError("Could not construct unique sealed questions")
    verify_new_exam(output)
    return output


def verify_new_exam(rows: list[RoleRow]) -> None:
    if (len(rows) != 288 or len({r.id for r in rows}) != len(rows)
            or len({r.prompt for r in rows}) != len(rows)):
        raise ValueError("Exam sample count or uniqueness invalid")
    for row in rows:
        truth = source_facts(row.template, row.story)
        if (truth["giver"] != row.giver or truth["recipient"] != row.recipient
                or truth["obj"] != row.object_name or row.giver == row.recipient):
            raise ValueError("Exam story facts mismatch independently parsed truth")
        if row.answer != truth[row.role]:
            raise ValueError("Exam gold answer differs from independent parser")
        qparts = row.prompt.split("\nQuestion: ")
        if len(qparts) != 2 or qparts[0] != "Story: " + row.story:
            raise ValueError("Exam source and question mismatch")
        allowed = [q.format(obj=truth["obj"]) + "\nAnswer:"
                   for q in EXAM_QUESTIONS[row.role]]
        if qparts[1] not in allowed:
            raise ValueError("Question semantics corrupted")
        candidates = candidate_spans(row.prompt)
        if {n for n, _, _ in candidates} != {row.giver, row.recipient}:
            raise ValueError("Entity detector failed to find both names")


def extended_train() -> list[RoleRow]:
    base = make_rows("train")
    output: list[RoleRow] = []
    for augmentation in range(len(TRAIN_QUESTIONS["giver"])):
        for row in base:
            question = TRAIN_QUESTIONS[row.role][augmentation].format(
                obj=row.object_name)
            output.append(replace(
                row,
                id=f"aug{augmentation}:{row.id}",
                trial=row.trial + 100 * augmentation,
                prompt=f"Story: {row.story}\nQuestion: {question}\nAnswer:",
            ))
    if len({r.prompt for r in output}) != len(output):
        raise ValueError("Training question augmentation produced exact duplicates")
    if len(output) != len(base) * len(TRAIN_QUESTIONS["giver"]):
        raise ValueError("Unexpected training question count")
    return output


class TwoStageHead(nn.Module):
    """A learnable story->giver scorer and question->role classifier.

    The forward method receives only story entity representations and
    question *text features*, never gold giver/recipient or the correct
    answer. Their independently learned probabilities are composed.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.story_ranker = nn.Sequential(
            nn.Linear(2 * width + 9, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Linear(width, 1),
        )
        self.query_ngrams = nn.Embedding(2049, width, padding_idx=0)
        self.query_classifier = nn.Sequential(
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 2),
        )

    def forward(self, item: Feature, *, ablate: bool = False):
        slots = item.slot
        if slots.entity.shape != (2, 2 * self.width + 9):
            raise ValueError("Two independent story candidates are required")
        # Story-role prediction cannot see the question text.
        story_logits = self.story_ranker(slots.entity).flatten()
        question_embed = (
            torch.zeros(self.width, dtype=slots.entity.dtype)
            if ablate else self.query_ngrams(item.grams).mean(0)
        )
        question_logits = self.query_classifier(question_embed)
        story_logp = F.log_softmax(story_logits, dim=0)
        question_logp = F.log_softmax(question_logits, dim=0)
        # P(answer=0) = P(asked-giver)*P(giver=0)
        #             + P(asked-recipient)*P(giver=1)
        answer0 = torch.logsumexp(torch.stack((
            question_logp[0] + story_logp[0],
            question_logp[1] + story_logp[1],
        )), dim=0)
        answer1 = torch.logsumexp(torch.stack((
            question_logp[0] + story_logp[1],
            question_logp[1] + story_logp[0],
        )), dim=0)
        return story_logits, question_logits, torch.stack((answer0, answer1))


def auxiliary_loss(model: TwoStageHead, item: Feature,
                   *, false_giver: bool = False, ablate: bool = False) -> torch.Tensor:
    slot = item.slot
    giver_label = slot.names.index(slot.answer) if slot.role == "giver" else (
        1 - slot.names.index(slot.answer))
    role_label = 0 if slot.role == "giver" else 1
    if false_giver:
        giver_label = 1 - giver_label
    story, question, answer = model(item, ablate=ablate)
    target = giver_label if slot.role == "giver" else 1 - giver_label
    # Three separate signals; question supervision uses label ONLY in loss.
    return (F.cross_entropy(story[None, :], torch.tensor([giver_label]))
            + F.cross_entropy(question[None, :], torch.tensor([role_label]))
            + 0.25 * F.nll_loss(answer[None, :], torch.tensor([target])))


@torch.no_grad()
def evaluate_two_stage(model: TwoStageHead, rows: list[Feature],
                       *, ablate: bool = False) -> dict:
    model.eval()
    cache = {}
    query_correct = 0
    story_correct = 0
    for item in rows:
        story, question, answer = model(item, ablate=ablate)
        cache[item.slot.id] = item.slot.names[int(answer.argmax())]
        q_gold = 0 if item.slot.role == "giver" else 1
        query_correct += int(question.argmax().item() == q_gold)
        giver_gold = item.slot.names.index(item.slot.answer) if q_gold == 0 else (
            1 - item.slot.names.index(item.slot.answer))
        story_correct += int(story.argmax().item() == giver_gold)
    scored, predictions = score_rows(
        [item.slot for item in rows], lambda row: cache[row.id])
    pairs = groups(rows)
    changed = sum(
        predictions[rows[p[0]].slot.id] != predictions[rows[p[1]].slot.id]
        for p in pairs
    )
    scored["question_role_accuracy"] = {
        "correct": query_correct, "total": len(rows)}
    scored["story_giver_accuracy"] = {
        "correct": story_correct, "total": len(rows)}
    scored["changed_answer_when_question_changed"] = {
        "correct": changed, "total": len(pairs)}
    return scored


@torch.no_grad()
def evaluate_monolithic(model: ContrastRoleHead, rows: list[Feature]) -> dict:
    return measure(model, rows)


def train(*, seed: int, training: list[Feature], dev: list[Feature],
          exam: list[Feature], steps: int, max_seconds: float,
          checkpoint_dir: Path | None = None,
          eval_every: int = 250) -> dict:
    if seed not in SEEDS or not 1 <= steps <= 10000 or not 1 <= eval_every <= steps:
        raise ValueError("Unregistered training configuration")
    if not 10 <= max_seconds <= 600:
        raise ValueError("Invalid CPU training time limit")
    pair_idx = groups(training)
    null_targets = group_flip_null(training, pair_idx, seed + 321)
    rng = random.Random(seed + 109)
    torch.manual_seed(seed)
    original = TwoStageHead(training[0].slot.query.numel())
    heads = {
        "factorized": copy.deepcopy(original),
        "wrong_story": copy.deepcopy(original),
        "question_off": copy.deepcopy(original),
        "monolithic": ContrastRoleHead(training[0].slot.query.numel()),
    }
    optimizer = {name: torch.optim.AdamW(net.parameters(), lr=0.0008)
                 for name, net in heads.items()}
    # Identical schedule and updates per arm; monolithic uses same number
    # of input events but has different capacity and a distinct objective.
    begun = time.monotonic()
    deadline = begun + max_seconds
    best = {}
    snapshots = []
    completed = 0
    for step in range(1, steps + 1):
        if time.monotonic() > deadline - 3:
            break
        p = pair_idx[rng.randrange(len(pair_idx))]
        for arm, net in heads.items():
            optim = optimizer[arm]
            optim.zero_grad(set_to_none=True)
            if arm == "monolithic":
                labels = [x.slot.gold for x in training]
                loss = pair_loss(net, training, p, labels,
                                 ablate=False, contrastive=True)
            else:
                loss = torch.stack([
                    auxiliary_loss(
                        net, training[i], false_giver=(
                            arm == "wrong_story" and
                            null_targets[i] != training[i].slot.gold),
                        ablate=arm == "question_off")
                    for i in p
                ]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optim.step()
        completed = step
        if step == 1 or step % eval_every == 0 or step == steps:
            snapshot = {"step": step, "elapsed": round(time.monotonic() - begun, 2),
                        "dev": {}}
            for arm, net in heads.items():
                result = (evaluate_monolithic(net, dev)
                          if arm == "monolithic" else evaluate_two_stage(
                              net, dev, ablate=arm == "question_off"))
                # Only *development* scores decide best saved weights.
                selector = (
                    result["both_questions_correct"]["correct"],
                    result["correct"],
                )
                snapshot["dev"][arm] = {
                    "both": selector[0], "exact": selector[1]}
                if arm not in best or selector > best[arm]["selector"]:
                    best[arm] = {
                        "selector": selector, "step": step,
                        "weights": copy.deepcopy(net.state_dict()),
                    }
            snapshots.append(snapshot)
            print(json.dumps({"seed": seed, **snapshot}), flush=True)
            if checkpoint_dir is not None:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                dst = checkpoint_dir / f"seed-{seed}-research.pt"
                tmp = dst.with_suffix(".tmp")
                torch.save({
                    "schema": 1, "experimental_only": True,
                    "seed": seed, "step": step, "best": best,
                    "state": {k: v.state_dict() for k, v in heads.items()},
                    "optim": {k: v.state_dict() for k, v in optimizer.items()},
                    "python_rng": rng.getstate(),
                    "torch_rng": torch.get_rng_state(),
                }, tmp)
                os.replace(tmp, dst)
    if completed == 0:
        raise RuntimeError("Training deadline expired before any updates")
    report = {"seed": seed, "steps": completed,
              "elapsed_seconds": round(time.monotonic() - begun, 2),
              "arms": {}, "development_curve": snapshots}
    for arm, net in heads.items():
        net.load_state_dict(best[arm]["weights"])
        d = (evaluate_monolithic(net, dev)
             if arm == "monolithic" else evaluate_two_stage(
                 net, dev, ablate=arm == "question_off"))
        # The sealed exam is scored ONCE, after development-only selection.
        e = (evaluate_monolithic(net, exam)
             if arm == "monolithic" else evaluate_two_stage(
                 net, exam, ablate=arm == "question_off"))
        report["arms"][arm] = {
            "best_dev_step": best[arm]["step"],
            "dev": d, "unseen_synthetic_exam": e,
        }
    return report


def experiment(*, source: Path, parent: Path, pointer: Path,
               out: Path, steps: int = 1600,
               max_minutes: float = 5.0, seeds: int = 2) -> dict:
    if not 1 <= steps <= 10000 or not 0.5 <= max_minutes <= 8 or seeds not in (1, 2):
        raise ValueError("Invalid bounded CPU pilot")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    verify_disjoint()
    audit_exam_vocab()
    raw_train = extended_train()
    raw_dev = make_rows("dev")
    raw_exam = make_new_exam()
    if ({r.prompt for r in raw_train} & {r.prompt for r in raw_exam}
            or {r.prompt for r in raw_dev} & {r.prompt for r in raw_exam}):
        raise ValueError("The sealed exam leaked into training")
    backbone, codec, _, lineage = load_frozen_models(source, parent, pointer)
    training = features(backbone, codec, raw_train)
    dev = features(backbone, codec, raw_dev)
    final = features(backbone, codec, raw_exam)
    out.mkdir(parents=True, exist_ok=True)
    begin = time.monotonic()
    results = []
    for i, seed in enumerate(SEEDS[:seeds]):
        remaining = max_minutes * 60 - (time.monotonic() - begin)
        if remaining < 20:
            break
        each = remaining / (seeds - i)
        results.append(train(
            seed=seed, training=training, dev=dev, exam=final,
            steps=steps, max_seconds=min(600, max(10, each)),
            checkpoint_dir=out / "recovery",
            eval_every=min(250, steps),
        ))
    report = {
        "schema": 1, "status": "experimental-only", "promoted": False,
        "source_lineage": lineage,
        "frozen_base": all(not p.requires_grad for p in backbone.parameters()),
        "new_exam_is_synthetic_not_natural": True,
        "seeds_requested": seeds, "seeds_completed": len(results),
        "counts": {"train": len(training), "dev": len(dev), "exam": len(final)},
        "hashes": {
            key: hashlib.sha256("\n".join(row.prompt for row in rows).encode()).hexdigest()
            for key, rows in (("train", raw_train), ("dev", raw_dev), ("exam", raw_exam))
        },
        "head_parameters": {
            "two_stage": sum(p.numel() for p in TwoStageHead(
                training[0].slot.query.numel()).parameters()),
            "monolithic": sum(p.numel() for p in ContrastRoleHead(
                training[0].slot.query.numel()).parameters()),
        },
        "budget_minutes": max_minutes, "requested_steps_per_seed": steps,
        "elapsed_seconds": round(time.monotonic() - begin, 2),
        "peak_resident_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "results": results,
        "limits": [
            "Synthetic two-person tasks with deterministic capitalization heuristic",
            "Question paraphrases are manually specified and not language-wide knowledge",
            "The old monolithic arm has different capacity and a contrastive objective",
            "Wrong-story control randomly flips entire paired events while preserving roles and answer balance",
            "A positive synthetic score does not establish natural-language understanding",
            "Original scratch-language weights are unchanged; no automatic promotion",
            "Original public datasets and user-confirmed personal memory stay separate",
        ],
    }
    (out / "factorized-v3-results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "seeds_completed": len(results),
        "elapsed": report["elapsed_seconds"],
        "exams": [{
            "seed": row["seed"], "steps": row["steps"],
            "results": {name: {
                "exact": arm["unseen_synthetic_exam"]["correct"],
                "pairs": arm["unseen_synthetic_exam"]["both_questions_correct"],
            } for name, arm in row["arms"].items()}
        } for row in results],
    }, indent=2), flush=True)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--pointer", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results/v3"))
    p.add_argument("--steps", type=int, default=1600)
    p.add_argument("--max-minutes", type=float, default=5)
    p.add_argument("--seeds", type=int, choices=(1, 2), default=2)
    a = p.parse_args()
    experiment(source=a.source, parent=a.parent, pointer=a.pointer,
               out=a.out, steps=a.steps,
               max_minutes=a.max_minutes, seeds=a.seeds)


if __name__ == "__main__":
    main()
