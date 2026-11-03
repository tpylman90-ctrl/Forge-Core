"""Forge role-contrast v2: original scratch-model backbone, matched controls.

Three predeclared seeds, no canonical-weight updates, CPU wall-clock caps.
A final synthetic exam uses NEW names, objects and grammar, and is scored
only after training/selection on a separate development split.

This is controlled semantic-role research, NOT evidence of natural language
understanding or a license to collect personal data.
"""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from dataclasses import dataclass
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
from forge_core.neural.role_conditioned_v1 import (
    Slots, score_rows, nearby_cues,
)
from forge_core.neural.span_pointer_lab import frozen_hidden
from forge_core.neural.role_curriculum_v1 import (
    RoleRow, candidate_spans, make_rows, verify_disjoint,
)

SEEDS = (7382, 7383, 7384)
EXAM_SEEDS = (10901, 10902, 10903)
EXAM_NAMES = ("Talena", "Veldon", "Isolde", "Cyril", "Marae",
              "Safiya", "Amon", "Elric", "Wren", "Ivara", "Zola",
              "Maddox", "Fiora", "Kelvin", "Selma", "Halim")
EXAM_OBJECTS = ("goblet", "trowel", "pendant", "hourglass", "glider",
                "dossier", "medallion", "harp", "mosaic", "prism")
EXAM_SYNTAX = (
    ("square", "in the square, the {obj} passed out of {giver}'s hands and into {recipient}'s.",
     r"in the square, the (?P<obj>[a-z]+) passed out of (?P<giver>[A-Z][a-z]+)'s hands and into (?P<recipient>[A-Z][a-z]+)'s\."),
    ("trail", "on the trail, {recipient} acquired the {obj}, which {giver} had brought.",
     r"on the trail, (?P<recipient>[A-Z][a-z]+) acquired the (?P<obj>[a-z]+), which (?P<giver>[A-Z][a-z]+) had brought\."),
    ("station", "at the station, {giver} transferred ownership of the {obj} to {recipient}.",
     r"at the station, (?P<giver>[A-Z][a-z]+) transferred ownership of the (?P<obj>[a-z]+) to (?P<recipient>[A-Z][a-z]+)\."),
)
EXAM_QUESTIONS = {
    "giver": ("Who originally supplied the {obj}?",
              "Which person was the source of the {obj}?"),
    "recipient": ("Who ended up owning the {obj}?",
                  "Which person took possession of the {obj}?"),
}
# New frozen exam is not used for early stopping, learning-rate selection,
# checkpoint promotion, or selecting which model head to deploy.


def _prior_vocab() -> tuple[set[str], set[str]]:
    from forge_core.neural.preschool_copy import (
        TRAIN_PEOPLE, EVAL_PEOPLE, TRAIN_WORDS, EVAL_WORDS,
    )
    from forge_core.neural.pointer_transfer import EXAM_PEOPLE, EXAM_WORDS
    from forge_core.neural.role_binding_diagnostic import (
        PEOPLE as OLD_PEOPLE, OBJECTS as OLD_OBJECTS,
    )
    from forge_core.neural.role_curriculum_v1 import NAMES, OBJECTS
    names = set(TRAIN_PEOPLE) | set(EVAL_PEOPLE) | set(EXAM_PEOPLE) | set(OLD_PEOPLE)
    words = set(TRAIN_WORDS) | set(EVAL_WORDS) | set(EXAM_WORDS) | set(OLD_OBJECTS)
    for group in NAMES.values():
        names.update(group)
    for group in OBJECTS.values():
        words.update(group)
    return names, words


def audit_exam_vocab() -> None:
    old_names, old_words = _prior_vocab()
    if (set(EXAM_NAMES) & old_names or set(EXAM_OBJECTS) & old_words
            or len(EXAM_NAMES) != len(set(EXAM_NAMES))
            or len(EXAM_OBJECTS) != len(set(EXAM_OBJECTS))):
        raise ValueError("Final role exam vocabulary overlaps previous datasets")


def make_exam() -> list[RoleRow]:
    audit_exam_vocab()
    rows: list[RoleRow] = []
    seen: set[str] = set()
    for seed in EXAM_SEEDS:
        rng = random.Random(seed)
        for name, pattern, grammar in EXAM_SYNTAX:
            for trial in range(10):
                for _attempt in range(128):
                    a, b = rng.sample(EXAM_NAMES, 2)
                    obj = rng.choice(EXAM_OBJECTS)
                    pending: list[RoleRow] = []
                    for swap in (0, 1):
                        giver, recipient = ((a, b) if swap == 0 else (b, a))
                        story = pattern.format(giver=giver, recipient=recipient, obj=obj)
                        parsed = re.fullmatch(grammar, story)
                        if not parsed:
                            raise ValueError("Final exam story did not pass independent parser")
                        facts = parsed.groupdict()
                        if (facts["giver"] != giver or facts["recipient"] != recipient
                                or facts["obj"] != obj):
                            raise ValueError("Independent gold differs from constructed facts")
                        for role in ("giver", "recipient"):
                            question = EXAM_QUESTIONS[role][trial % 2].format(obj=obj)
                            prompt = f"Story: {story}\nQuestion: {question}\nAnswer:"
                            pending.append(RoleRow(
                                id=f"exam-v2:{seed}:{name}:{trial}:{swap}:{role}",
                                split="exam-v2", template=name, story=story,
                                prompt=prompt, answer=facts[role], role=role,
                                giver=facts["giver"], recipient=facts["recipient"],
                                object_name=obj, seed=seed, trial=trial, swap=swap,
                            ))
                    if len({r.prompt for r in pending}) == 4 and not (
                            {r.prompt for r in pending} & seen):
                        rows.extend(pending)
                        seen.update(r.prompt for r in pending)
                        break
                else:
                    raise ValueError("Cannot generate non-overlapping final exam questions")
    audit_exam(rows)
    return rows


def audit_exam(rows: list[RoleRow]) -> None:
    if len(rows) != 360 or len({r.id for r in rows}) != len(rows):
        raise ValueError("Incorrect final role exam case count")
    if len({r.prompt for r in rows}) != len(rows):
        raise ValueError("Duplicate final role exam prompts")
    grammar_by_name = {name: grammar for name, _, grammar in EXAM_SYNTAX}
    for row in rows:
        grammar = grammar_by_name.get(row.template)
        if grammar is None:
            raise ValueError("Unregistered final grammar")
        match = re.fullmatch(grammar, row.story)
        if match is None:
            raise ValueError("Final story failed independent grammar")
        parsed = match.groupdict()
        if (row.giver != parsed["giver"] or row.recipient != parsed["recipient"]
                or row.object_name != parsed["obj"] or row.giver == row.recipient):
            raise ValueError("Final event facts differ from parsed source")
        expected = parsed[row.role]
        if row.answer != expected:
            raise ValueError("Final gold answer differs from parsed source")
        q = row.prompt.split("\nQuestion: ", 1)
        if len(q) != 2 or q[0] != "Story: " + row.story:
            raise ValueError("Final question source layout altered")
        legal = [v.format(obj=parsed["obj"]) + "\nAnswer:"
                 for v in EXAM_QUESTIONS[row.role]]
        if q[1] not in legal:
            raise ValueError("Final question does not match its semantic role")
        if {name for name, _, _ in candidate_spans(row.prompt)} != {
                parsed["giver"], parsed["recipient"]}:
            raise ValueError("Unverified candidate identification")


def question_grams(prompt: str) -> torch.Tensor:
    """Stable hashed character 3-grams, from the QUESTION alone.

    Hashes are features, not pretrained embeddings. Never include the
    source answer or verified role labels; all embedding weights start random.
    """
    if prompt.count("\nQuestion: ") != 1 or not prompt.endswith("\nAnswer:"):
        raise ValueError("Unexpected question layout")
    q = prompt.split("\nQuestion: ", 1)[1][:-len("\nAnswer:")].lower()
    q = re.sub(r"[^a-z ]", " ", q).strip()
    if len(q) < 3:
        raise ValueError("Question too short")
    padded = " " + " ".join(q.split()) + " "
    grams = [1 + int.from_bytes(hashlib.blake2b(
        padded[i:i + 3].encode(), digest_size=4).digest(), "little") % 2048
             for i in range(len(padded) - 2)]
    return torch.tensor(grams[:120], dtype=torch.long)


@dataclass
class Feature:
    slot: Slots
    grams: torch.Tensor


@torch.no_grad()
def story_only_slots(backbone, codec, row: RoleRow) -> Slots:
    """Candidate representations cannot encode the later question.

    The source name embeddings may attend to earlier story tokens only.
    The optional three-token suffix must be clamped before the Question
    marker, preventing a partially ablated baseline from seeing the query.
    """
    enc = codec.encode(row.prompt)
    hidden = frozen_hidden(backbone, enc.ids)
    boundary = row.prompt.index("\nQuestion: ")
    story_end = max(i for i, (left, right) in enumerate(enc.offsets)
                    if right <= boundary)
    spans = candidate_spans(row.prompt)
    vectors = []
    for index, (name, start, end) in enumerate(spans):
        covered = [i for i, (left, right) in enumerate(enc.offsets)
                   if right > start and left < end and i <= story_end]
        if not covered:
            raise ValueError("Entity could not map to story tokens")
        after = min(max(covered) + 3, story_end)
        local_left, local_right = start - 7, end - 7
        metadata = [
            float(index == 0), float(index == 1),
            float(local_left / max(1, len(row.story))),
            *nearby_cues(row.story, local_left, local_right),
        ]
        if len(metadata) != 9:
            raise ValueError("Unexpected nongold candidate features")
        vectors.append(torch.cat((
            hidden[covered].mean(dim=0),
            hidden[after],
            torch.tensor(metadata, dtype=hidden.dtype),
        )))
    names = tuple(name for name, _, _ in spans)
    if row.answer not in names:
        raise ValueError("Verified answer missing from independent entity detector")
    return Slots(row.id, row.role, row.template, row.seed, row.trial,
                 row.swap, row.answer, names, hidden[-1],
                 torch.stack(vectors), names.index(row.answer))


def features(backbone, codec, rows: list[RoleRow]) -> list[Feature]:
    return [Feature(story_only_slots(backbone, codec, row),
                    question_grams(row.prompt)) for row in rows]


def groups(rows: list[Feature]) -> list[tuple[int, int, int, int]]:
    bucket: dict[tuple[int, str, int], dict[tuple[int, str], int]] = defaultdict(dict)
    for i, item in enumerate(rows):
        r = item.slot
        key = (r.seed, r.template, r.trial)
        if (r.swap, r.role) in bucket[key]:
            raise ValueError("Repeated paired role question")
        bucket[key][(r.swap, r.role)] = i
    result = []
    for key in sorted(bucket):
        pair = bucket[key]
        if set(pair) != {(0, "giver"), (0, "recipient"), (1, "giver"), (1, "recipient")}:
            raise ValueError("Unpaired role/swap contrast")
        indexes = (pair[(0, "giver")], pair[(0, "recipient")],
                   pair[(1, "giver")], pair[(1, "recipient")])
        for a, b in ((0, 1), (2, 3), (0, 2), (1, 3)):
            if rows[indexes[a]].slot.answer == rows[indexes[b]].slot.answer:
                raise ValueError("Counterfactual answers do not oppose")
        result.append(indexes)
    return result


def group_flip_null(rows: list[Feature], pairs: list[tuple[int, ...]],
                    seed: int) -> list[int]:
    """A coherent but semantically wrong role-label control.

    Each four-question event family is flipped as a unit, preserving paired
    label structure, answer-position balance and the same optimizer budget.
    """
    rng = random.Random(seed)
    targets = [r.slot.gold for r in rows]
    flipped = 0
    for pair in pairs:
        if rng.randrange(2):
            flipped += 1
            for i in pair:
                targets[i] = 1 - targets[i]
    if not 0 < flipped < len(pairs):
        raise ValueError("Degenerate shuffled event-label control")
    return targets


class ContrastRoleHead(nn.Module):
    """Trainable, from-scratch question encoder and entity interaction."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.entity_projection = nn.Sequential(
            nn.Linear(width * 2 + 9, width),
            nn.GELU(), nn.LayerNorm(width),
        )
        self.query_projection = nn.Linear(width, width)
        self.ngram = nn.Embedding(2049, width, padding_idx=0)
        self.question_norm = nn.LayerNorm(width)
        self.scorer = nn.Sequential(
            nn.Linear(width * 4, width),
            nn.Tanh(), nn.Linear(width, 1),
        )

    def forward(self, feature: Feature, *, ablate_question: bool = False) -> torch.Tensor:
        s = feature.slot
        if s.entity.shape != (2, self.width * 2 + 9):
            raise ValueError("Two independently detected candidate names required")
        if s.query.shape != (self.width,):
            raise ValueError("Invalid frozen backbone question shape")
        entity = self.entity_projection(s.entity)
        if ablate_question:
            # Constant zero vector, not a learned role label or query text.
            question = torch.zeros_like(s.query)
            grams = torch.tensor([0], dtype=torch.long)
        else:
            question = s.query
            grams = feature.grams
        embedding = self.ngram(grams).mean(dim=0)
        q = self.question_norm(torch.tanh(
            self.query_projection(question) + embedding))
        q = q.unsqueeze(0).expand_as(entity)
        interaction = torch.cat((entity, q, entity * q, torch.abs(entity - q)), dim=1)
        return self.scorer(interaction).flatten()


@torch.no_grad()
def measure(head: ContrastRoleHead, rows: list[Feature],
            *, ablate_question: bool = False) -> dict:
    head.eval()
    mapping = {r.slot.id: r for r in rows}
    scored, predictions = score_rows([r.slot for r in rows], lambda s:
        s.names[int(head(mapping[s.id], ablate_question=ablate_question).argmax())])
    # Opposite questions must actually change the answer in order to pass.
    pairs = groups(rows)
    disagree = sum(
        predictions[rows[p[0]].slot.id] != predictions[rows[p[1]].slot.id]
        for p in pairs
    )
    scored["changed_answer_when_question_role_changed"] = {
        "correct": disagree, "total": len(pairs)}
    return scored


def pair_loss(head: ContrastRoleHead, rows: list[Feature], pair: tuple[int, ...],
              targets: list[int], *, ablate: bool, contrastive: bool) -> torch.Tensor:
    scores = [head(rows[i], ablate_question=ablate) for i in pair]
    ce = torch.stack([
        F.cross_entropy(logits[None, :], torch.tensor([targets[i]]))
        for i, logits in zip(pair, scores)
    ]).mean()
    if not contrastive:
        return ce
    # Explicitly demand opposing answers for giver vs recipient questions.
    # Using the same answer for both questions incurs a pairwise penalty.
    contrast_terms = []
    for a, b in ((0, 1), (2, 3)):
        if targets[pair[a]] == targets[pair[b]]:
            raise ValueError("Contradictory role pair does not have opposing targets")
        a_target = targets[pair[a]]
        margin_a = scores[a][a_target] - scores[a][1 - a_target]
        margin_b = scores[b][a_target] - scores[b][1 - a_target]
        contrast_terms.append(F.softplus(0.7 - (margin_a - margin_b)))
    return ce + 0.4 * torch.stack(contrast_terms).mean()


def train(seed: int, *, rows: list[Feature], development: list[Feature],
          exam: list[Feature], max_steps: int, eval_every: int,
          deadline: float, lr: float = 0.0006,
          snapshot_dir: Path | None = None, source_hash: str = "") -> dict:
    if seed not in SEEDS:
        raise ValueError("Unregistered independent initialization")
    pair_indices = groups(rows)
    rng = random.Random(seed + 1900)
    shuffle_targets = group_flip_null(rows, pair_indices, seed + 251)
    gold = [row.slot.gold for row in rows]
    torch.manual_seed(seed)
    initial = ContrastRoleHead(rows[0].slot.query.numel())
    heads = {
        "paired": copy.deepcopy(initial),
        "cross_entropy": copy.deepcopy(initial),
        "question_ablated": copy.deepcopy(initial),
        "shuffled_pairs": copy.deepcopy(initial),
    }
    optimizers = {name: torch.optim.AdamW(h.parameters(), lr=lr)
                  for name, h in heads.items()}
    settings = {
        "paired": (gold, False, True),
        "cross_entropy": (gold, False, False),
        "question_ablated": (gold, True, True),
        "shuffled_pairs": (shuffle_targets, False, True),
    }
    # Choose heads strictly on DEVELOPMENT set; the final exam is never read
    # until training has stopped for the whole run.
    best_state = {}
    best_score = {}
    step = 0
    curve = []
    start = time.monotonic()
    for step in range(1, max_steps + 1):
        if time.monotonic() >= deadline - 5:
            step -= 1
            break
        selected = pair_indices[rng.randrange(len(pair_indices))]
        for name, head in heads.items():
            labels, ablate, contrasted = settings[name]
            loss = pair_loss(
                head, rows, selected, labels,
                ablate=ablate, contrastive=contrasted)
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizers[name].step()
        if step == 1 or step % eval_every == 0 or step == max_steps:
            snapshot = {}
            for name, head in heads.items():
                measured = measure(
                    head, development,
                    ablate_question=(name == "question_ablated"))
                key = (measured["both_questions_correct"]["correct"],
                       measured["correct"])
                snapshot[name] = {
                    "dev_correct": measured["correct"],
                    "dev_pairs_both_correct": measured["both_questions_correct"]["correct"],
                }
                if name not in best_score or key > best_score[name]:
                    best_score[name] = key
                    best_state[name] = {
                        "step": step,
                        "weights": copy.deepcopy(head.state_dict()),
                    }
            curve.append({
                "step": step, "elapsed_seconds": round(time.monotonic() - start, 2),
                "dev": snapshot,
            })
            print(json.dumps({"seed": seed, **curve[-1]}), flush=True)
            if snapshot_dir is not None:
                # Recovery/research snapshot. No automatic model promotion.
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                archive = {
                    "schema": 2, "experimental_only": True,
                    "seed": seed, "step": step,
                    "source_hash": source_hash,
                    "model_weights": {n: h.state_dict() for n, h in heads.items()},
                    "optimizers": {n: o.state_dict() for n, o in optimizers.items()},
                    "best_by_development": best_state,
                    "dev_curve": curve,
                    "python_sampler_state": rng.getstate(),
                    "torch_rng_state": torch.get_rng_state(),
                }
                target = snapshot_dir / f"seed-{seed}-recovery.pt"
                tmp = target.with_suffix(".tmp")
                torch.save(archive, tmp)
                os.replace(tmp, target)
            if time.monotonic() >= deadline - 5:
                break

    # Select using development only; now evaluate the final synthetic exam once.
    output = {"seed": seed, "updates_per_arm": step,
              "elapsed_seconds": round(time.monotonic() - start, 2),
              "dev_curve": curve, "heads": {}}
    for name, head in heads.items():
        head.load_state_dict(best_state[name]["weights"])
        heads[name].eval()
        output["heads"][name] = {
            "best_development_step": best_state[name]["step"],
            "development": measure(head, development,
                                   ablate_question=name == "question_ablated"),
            "final_synthetic_exam": measure(
                head, exam, ablate_question=name == "question_ablated"),
        }
    output["all_heads_random_initialized"] = True
    output["original_scratch_backbone_unchanged"] = True
    return output


def experiment(*, source: Path, parent: Path, pointer: Path, out: Path,
               max_minutes: float = 32.0, max_steps: int = 12000,
               eval_every: int = 300, quick: bool = False) -> dict:
    if not 0.1 <= max_minutes <= 35.0 or not 1 <= max_steps <= 50000:
        raise ValueError("Invalid bounded CPU experiment budget")
    if not 1 <= eval_every <= max_steps:
        raise ValueError("Invalid development evaluation cadence")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    verify_disjoint()
    source_rows = make_rows("train")
    dev_rows = make_rows("dev")
    final_rows = make_exam()
    if ({r.prompt for r in source_rows} & {r.prompt for r in final_rows}
            or {r.prompt for r in dev_rows} & {r.prompt for r in final_rows}):
        raise ValueError("Synthetic final exam leaked into prior curriculum")
    start = time.monotonic()
    backbone, codec, _heads, lineage = load_frozen_models(source, parent, pointer)
    train_features = features(backbone, codec, source_rows)
    dev_features = features(backbone, codec, dev_rows)
    exam_features = features(backbone, codec, final_rows)
    names = list(SEEDS if not quick else SEEDS[:1])
    train_hash = hashlib.sha256(
        "\n".join(r.prompt for r in source_rows).encode()).hexdigest()
    deadline = time.monotonic() + max_minutes * 60
    results = []
    for index, seed in enumerate(names):
        left = len(names) - index
        remaining = max(0, deadline - time.monotonic())
        slice_deadline = time.monotonic() + remaining / left
        if remaining < 30:
            break
        results.append(train(
            seed, rows=train_features, development=dev_features,
            exam=exam_features, max_steps=max_steps, eval_every=eval_every,
            deadline=slice_deadline, snapshot_dir=out / "recovery",
            source_hash=train_hash,
        ))
    report = {
        "schema": 1, "mode": "contrastive-roles-and-question-ablation",
        "research_only": True, "not_promoted": True,
        "source_lineage": lineage,
        "params_per_head": sum(p.numel() for p in ContrastRoleHead(
            train_features[0].slot.query.numel()).parameters()),
        "seeds": names, "finished_seeds": len(results),
        "data_counts": {"train": len(source_rows), "development": len(dev_rows),
                        "new_final_synthetic_exam": len(final_rows)},
        "data_hashes": {
            split: hashlib.sha256("\n".join(r.prompt for r in rows).encode()).hexdigest()
            for split, rows in (("train", source_rows),
                                ("development", dev_rows), ("exam", final_rows))
        },
        "controls": [
            "paired contrasts", "same-data cross-entropy",
            "question-ablated paired contrasts",
            "group-flipped null label pairs",
        ],
        "budget_minutes": max_minutes,
        "max_steps_per_seed_per_arm": max_steps,
        "total_elapsed_seconds": round(time.monotonic() - start, 2),
        "updates_per_second": round(
            sum(r["updates_per_arm"] for r in results) /
            max(1, sum(r["elapsed_seconds"] for r in results)), 3),
        "peak_resident_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "results": results,
        "limitations": [
            "Externally unverified synthetic two-person rules and source-gold answers",
            "Both previous training and development grammar families were reused",
            "Exact final templates are new but not independent natural-language annotation",
            "No attempt to promote any candidate or change archived original Forge weights",
            "Question ablation removes both frozen query and lexical question features",
            "CPU timings do not establish any official GitHub daily allowance",
            "Recovery snapshots preserve optimizer/RNG state but automated resumption is not yet integrated",
            "No model is evidence of why an actual human made a decision",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "contrastive-v2-report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "finished_seeds": len(results),
        "budget_minutes": max_minutes,
        "total_elapsed_seconds": report["total_elapsed_seconds"],
        "updates_per_second": report["updates_per_second"],
        "exams": [{
            "seed": r["seed"],
            "updates": r["updates_per_arm"],
            "paired_both_correct": r["heads"]["paired"]["final_synthetic_exam"]["both_questions_correct"],
            "ce_both_correct": r["heads"]["cross_entropy"]["final_synthetic_exam"]["both_questions_correct"],
            "ablated_both_correct": r["heads"]["question_ablated"]["final_synthetic_exam"]["both_questions_correct"],
            "null_both_correct": r["heads"]["shuffled_pairs"]["final_synthetic_exam"]["both_questions_correct"],
        } for r in results],
    }, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--pointer", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/role-contrast-v2"))
    parser.add_argument("--max-minutes", type=float, default=32)
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    experiment(source=args.source, parent=args.parent, pointer=args.pointer,
               out=args.out, max_minutes=args.max_minutes,
               max_steps=args.max_steps, eval_every=args.eval_every,
               quick=args.quick)


if __name__ == "__main__":
    main()
