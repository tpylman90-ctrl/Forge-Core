"""Role-conditioned entity-slot pointer on Forge's FROZEN scratch backbone.

The model receives a natural-language question representation and two
candidate name slots detected WITHOUT consulting gold. A small, randomly
initialized neural scorer learns to choose the giver or recipient.

Matched shuffled-label controls and new template/name splits guard against
mistaking rote training for semantic understanding. NEVER auto-promote.
"""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from forge_core.neural.role_binding_diagnostic import load_frozen_models
from forge_core.neural.role_curriculum_v1 import (
    RoleRow, candidate_spans, make_rows, verify_disjoint,
)
from forge_core.neural.span_pointer_lab import features_for

CUE_WORDS = ("to", "from", "by")
EXTRA = 9  # first/second position, normalized char offset, 3 preceding, 3 following


@dataclass
class Slots:
    id: str
    role: str
    template: str
    seed: int
    trial: int
    swap: int
    answer: str
    names: tuple[str, str]
    query: torch.Tensor
    entity: torch.Tensor
    gold: int


def nearby_cues(story: str, left: int, right: int) -> list[float]:
    # Position and literal prepositions are input features, NOT gold answers.
    # They supply a limited inductive bias for the small scratch model.
    before = re.findall(r"[a-z]+", story[:left].lower())[-3:]
    after = re.findall(r"[a-z]+", story[right:].lower())[:3]
    return [
        float(word in before) for word in CUE_WORDS
    ] + [
        float(word in after) for word in CUE_WORDS
    ]


@torch.no_grad()
def slot_features(backbone, codec, row: RoleRow) -> Slots:
    from forge_core.neural.span_pointer_lab import frozen_hidden
    enc = codec.encode(row.prompt)
    hidden = frozen_hidden(backbone, enc.ids)
    spans = candidate_spans(row.prompt)
    story = row.story
    vectors = []
    for index, (name, start, end) in enumerate(spans):
        overlapping = [i for i, (l, r) in enumerate(enc.offsets)
                       if r > start and l < end]
        if not overlapping:
            raise ValueError("Name cannot be mapped to original BPE tokens")
        near_after = min(max(overlapping) + 3, len(enc.ids) - 1)
        relative_left, relative_right = start - 7, end - 7
        metadata = [
            float(index == 0), float(index == 1),
            float(relative_left / max(1, len(story))),
            *nearby_cues(story, relative_left, relative_right),
        ]
        if len(metadata) != EXTRA:
            raise AssertionError("Entity-feature shape mismatch")
        vectors.append(torch.cat((
            hidden[overlapping].mean(dim=0),
            hidden[near_after],
            torch.tensor(metadata, dtype=hidden.dtype),
        ), dim=0))
    names = tuple(s[0] for s in spans)
    if len(names) != 2 or row.answer not in names:
        raise ValueError("Unverified target or missing copyable person")
    return Slots(row.id, row.role, row.template, row.seed, row.trial,
                 row.swap, row.answer, names, hidden[-1],
                 torch.stack(vectors), names.index(row.answer))


class RoleHead(nn.Module):
    """Question-conditioned entity cross-attention scorer.

    No truth label, role ID or input answer is passed to forward().
    Distinct natural-language questions generate distinct query vectors.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.entity_key = nn.Sequential(
            nn.Linear(2 * width + EXTRA, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        self.question = nn.Linear(width, width)
        self.score = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.Tanh(),
            nn.Linear(width, 1),
        )

    def forward(self, query: torch.Tensor, entity: torch.Tensor) -> torch.Tensor:
        if query.ndim != 1 or query.numel() != self.width:
            raise ValueError("Invalid question representation")
        if entity.shape != (2, self.width * 2 + EXTRA):
            raise ValueError("Expected two independently extracted entity slots")
        keys = self.entity_key(entity)
        condition = self.question(query).unsqueeze(0).expand_as(keys)
        return self.score(torch.cat((keys, condition), dim=-1)).flatten()


def score_rows(rows: list[Slots], predict) -> tuple[dict, dict[str, str]]:
    predictions = {row.id: predict(row) for row in rows}
    if len(predictions) != len(rows):
        raise ValueError("Duplicate score-case identifiers")
    per_role: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    per_template: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    per_seed: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    correct = 0
    for row in rows:
        if predictions[row.id] not in row.names:
            raise ValueError("Model output is not an independently detected entity")
        hit = int(predictions[row.id] == row.answer)
        correct += hit
        for counts, key in ((per_role, row.role),
                            (per_template, row.template),
                            (per_seed, str(row.seed))):
            counts[key][0] += hit
            counts[key][1] += 1

    lookup = {(r.seed, r.template, r.trial, r.swap, r.role): r for r in rows}
    role_pairs = 0
    role_both = 0
    swap_pairs = 0
    swap_both = 0
    for seed, template, trial in sorted({(r.seed, r.template, r.trial) for r in rows}):
        for swap in (0, 1):
            giver = lookup[(seed, template, trial, swap, "giver")]
            receiver = lookup[(seed, template, trial, swap, "recipient")]
            role_pairs += 1
            role_both += int(predictions[giver.id] == giver.answer and
                             predictions[receiver.id] == receiver.answer)
        for role in ("giver", "recipient"):
            a = lookup[(seed, template, trial, 0, role)]
            b = lookup[(seed, template, trial, 1, role)]
            swap_pairs += 1
            swap_both += int(predictions[a.id] == a.answer and
                             predictions[b.id] == b.answer)

    def table(counter):
        return {k: {"correct": v[0], "total": v[1],
                    "accuracy": round(v[0] / v[1], 4)}
                for k, v in sorted(counter.items())}
    return ({
        "correct": correct, "total": len(rows),
        "accuracy": round(correct / len(rows), 4),
        "by_role": table(per_role), "by_template": table(per_template),
        "by_seed": table(per_seed),
        "both_questions_correct": {"correct": role_both, "total": role_pairs},
        "both_entity_swaps_correct": {"correct": swap_both, "total": swap_pairs},
    }, predictions)


@torch.no_grad()
def evaluate(head: RoleHead, rows: list[Slots]) -> dict:
    head.eval()
    result, _ = score_rows(
        rows, lambda r: r.names[int(head(r.query, r.entity).argmax())])
    return result


def shuffled_labels(rows: list[Slots], seed: int) -> list[int]:
    """Within-role shuffling keeps 0/1 prevalence but destroys grammatical links.

    Shuffling within template would be INVALID: in these templates each
    grammatical form fixes the position of giver and recipient, so labels
    would remain unchanged and create a fake control.
    """
    targets = [r.gold for r in rows]
    randomizer = random.Random(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row.role].append(i)
    for indexes in groups.values():
        labels = [targets[i] for i in indexes]
        randomizer.shuffle(labels)
        for i, label in zip(indexes, labels):
            targets[i] = label
    return targets


def train_head(model: RoleHead, rows: list[Slots], labels: list[int],
               schedule: list[list[int]], lr: float) -> list[dict]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    curve = []
    for step, batch in enumerate(schedule, 1):
        loss = torch.stack([
            F.cross_entropy(
                model(rows[i].query, rows[i].entity)[None, :],
                torch.tensor([labels[i]], dtype=torch.long))
            for i in batch
        ]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in (1, len(schedule)) or step % 100 == 0:
            curve.append({"step": step, "loss": round(float(loss.detach()), 6)})
    model.eval()
    return curve


@torch.no_grad()
def old_pointer_restricted(head, backbone, codec, rows: list[RoleRow]) -> dict:
    """Fairer two-name option for the previously trained general word pointer.

    Not a matched-compute control and never retrained for this task.
    """
    prepared = []
    for row in rows:
        base = features_for(backbone, codec, SimpleNamespace(\n            id=row.id, skill=row.role, prompt=row.prompt, answer=row.answer))
        allowed = set(name for name, _, _ in candidate_spans(row.prompt))
        indices = [i for i, name in enumerate(base.candidates) if name in allowed]
        if len(indices) != 2:
            raise ValueError("Expected exactly two name spans in generic pointer")
        logits = head(base.query, base.words)
        index = max(indices, key=lambda i: float(logits[i]))
        prepared.append((row, base.candidates[index]))
    predictions = {row.id: prediction for row, prediction in prepared}
    # Convert the previous pointer's per-question outputs into comparable metrics.
    slots = [Slots(row.id, row.role, row.template, row.seed, row.trial,
                   row.swap, row.answer,
                   tuple(name for name, _, _ in candidate_spans(row.prompt)),
                   torch.empty(0), torch.empty(0), -1) for row in rows]
    return score_rows(slots, lambda r: predictions[r.id])[0]


def experiment(*, source: Path, parent: Path, pointer: Path, out: Path,
               steps: int = 350, seed: int = 7381, lr: float = 0.001,
               batch_size: int = 8) -> dict:
    if not 1 <= steps <= 900 or not 1 <= seed <= 1_000_000 or not 2 <= batch_size <= 16 or not 0 < lr <= 0.01:
        raise ValueError("Outside preregistered CPU research budget")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    verify_disjoint()
    train = make_rows("train")
    dev = make_rows("dev")
    # The exam is generated by fixed original rules before any optimizer updates.
    exam = make_rows("exam")
    if len({r.prompt for r in train + dev + exam}) != len(train + dev + exam):
        raise ValueError("Training/development/exam question overlap")
    started = time.monotonic()
    backbone, codec, pointer_heads, lineage = load_frozen_models(source, parent, pointer)
    train_features = [slot_features(backbone, codec, r) for r in train]
    dev_features = [slot_features(backbone, codec, r) for r in dev]
    exam_features = [slot_features(backbone, codec, r) for r in exam]
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        trained = RoleHead(backbone.cfg.n_embd)
        shuffled = copy.deepcopy(trained)
        untrained = copy.deepcopy(trained)
    order = random.Random(seed + 141)
    schedule = [[order.randrange(len(train)) for _ in range(batch_size)]
                for _ in range(steps)]
    wrong = shuffled_labels(train_features, seed + 81)
    curve_trained = train_head(
        trained, train_features, [r.gold for r in train_features], schedule, lr)
    curve_shuffled = train_head(shuffled, train_features, wrong, schedule, lr)
    results = {
        "untrained": {"dev": evaluate(untrained, dev_features),
                      "exam": evaluate(untrained, exam_features)},
        "verified": {"dev": evaluate(trained, dev_features),
                     "exam": evaluate(trained, exam_features)},
        "shuffled": {"dev": evaluate(shuffled, dev_features),
                     "exam": evaluate(shuffled, exam_features)},
        "old_pointer_name_filtered": {
            "exam": old_pointer_restricted(pointer_heads["verified"], backbone, codec, exam),
        },
    }
    report = {
        "schema": 1, "research_only": True, "not_promoted": True,
        "scratch_backbone_frozen": all(not p.requires_grad for p in backbone.parameters()),
        "source_lineage": lineage, "model": "question-conditioned-two-entity-neural-pointer",
        "new_head_parameters": sum(p.numel() for p in trained.parameters()),
        "candidate_extraction": "capitalized words from source story only; no gold or role label",
        "question_input": "original frozen neural hidden state, no gold role ID or answer supplied",
        "local_cues": "two positional, one normalized offset, six source-preposition indicators",
        "seed": seed, "steps_per_arm": steps, "batch_size": batch_size, "lr": lr,
        "case_counts": {"train": len(train), "dev": len(dev), "exam": len(exam)},
        "shuffle_control_preserves_role_but_breaks_template_answer_relationship": True,
        "train_data_sha256": hashlib.sha256(
            "\n".join(r.prompt for r in train).encode()).hexdigest(),
        "dev_data_sha256": hashlib.sha256(
            "\n".join(r.prompt for r in dev).encode()).hexdigest(),
        "exam_data_sha256": hashlib.sha256(
            "\n".join(r.prompt for r in exam).encode()).hexdigest(),
        "train_curve": curve_trained, "shuffled_curve": curve_shuffled,
        "results": results, "duration_seconds": round(time.monotonic() - started, 2),
        "limitations": [
            "Small original synthetic rules and two-person constrained source questions",
            "Question wording and grammar are limited; no real-world understanding claim",
            "Capitalization-based entity detection is explicit, not learned",
            "Original pointer has different training and candidate distribution",
            "Exam is internally generated and one run; no independent natural-language evidence",
            "No personal memories, external teacher weights, Android integration or promotion",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "role-conditioned-report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, head in (("verified", trained), ("shuffled", shuffled)):
        torch.save({
            "schema": 1, "experimental_only": True, "arm": name,
            "source_lineage": lineage,
            "seed": seed, "steps": steps, "width": backbone.cfg.n_embd,
            "model": head.state_dict(),
        }, out / f"{name}-role-head.pt")
    print(json.dumps({
        "case_counts": report["case_counts"],
        "verified_exam": results["verified"]["exam"],
        "shuffled_exam": results["shuffled"]["exam"],
        "old_pointer_name_filtered": results["old_pointer_name_filtered"]["exam"],
        "untrained_exam": results["untrained"]["exam"],
        "original_backbone_frozen": report["scratch_backbone_frozen"],
        "research_only": True,
    }, indent=2), flush=True)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--pointer", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("artifacts/role-conditioned"))
    p.add_argument("--steps", type=int, default=350)
    p.add_argument("--seed", type=int, default=7381)
    a = p.parse_args()
    experiment(source=a.source, parent=a.parent, pointer=a.pointer,
               out=a.out, steps=a.steps, seed=a.seed)


if __name__ == "__main__":
    main()
