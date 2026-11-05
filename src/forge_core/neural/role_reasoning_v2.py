"""Matched multi-seed test of structured role explanations and name spelling.

A deterministic teacher supplies verified giver/recipient role labels and a
short, checkable explanation. Two supervised students receive the same
counterfactual episodes: one generates the person's name, the other generates
the corresponding candidate letter. This separates role binding from name
spelling. Shuffled-role and equal-compute text-only controls are trained from
the same scratch-origin checkpoint. No candidate is promoted.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import re
import time

import torch

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.preschool_direct import DirectExample, answer_loss, encode, verify_source
from forge_core.neural.seed_v07 import generate_samples, heldout
from forge_core.neural.train import make_batch


TRAIN_PEOPLE = ("Ava", "Blake", "Celia", "Devin", "Esme", "Felix",
                "Gia", "Holden", "Isla", "Joel", "Keira", "Luca")
FINAL_PEOPLE = ("Margo", "Nico", "Oona", "Pascal", "Rhea", "Sol",
                "Tessa", "Uriah", "Vance", "Willa", "Xavier", "Yvonne")
TRAIN_OBJECTS = ("lantern", "trumpet", "thimble", "telescope",
                 "postcard", "kettle", "locket", "mitten")
FINAL_OBJECTS = ("cymbal", "diary", "acorn", "compass",
                 "flask", "garland", "spool", "teapot")
TRAIN_STORIES = (
    "{giver} gave the {item} to {recipient}.",
    "{recipient} got the {item} from {giver}.",
    "{giver} passed the {item} to {recipient}.",
    "The {item} was handed to {recipient} by {giver}.",
)
FINAL_SEEN_VERB_STORIES = (
    "{giver} handed {item} across to {recipient}.",
    "{recipient} ended up with {item} after {giver} gave it to them.",
)
FINAL_UNSEEN_VERB_STORIES = (
    "{giver} entrusted the {item} to {recipient}.",
    "The {item} was delivered by {giver} into {recipient}'s hands.",
)
TRAIN_QUESTIONS = {
    "giver": ("Who gave the {item}?", "Who sent on the {item}?"),
    "recipient": ("Who received the {item}?", "Who got the {item}?"),
}
FINAL_QUESTIONS = {
    "giver": ("Which person set the transfer in motion?",
              "Who sent the {item} on its way?"),
    "recipient": ("Which person got hold of the {item}?",
                  "Who took the {item}?"),
}
SKILLS = ("giver", "recipient")
ARMS = ("structured_names", "structured_choice", "shuffled_labels", "text_only")
STAGES = ((160, 250, 0.05, 2e-4),
          (800, 650, 0.15, 1e-4),
          (3200, 1600, 0.25, 6e-5))


@dataclass(frozen=True)
class Case:
    id: str
    episode: str
    stratum: str
    skill: str
    giver: str
    recipient: str
    item: str
    story: str
    question: str
    option_a: str
    option_b: str
    name_answer: str
    letter_answer: str

    def prompt(self, mode: str) -> str:
        if mode not in ("name", "letter"):
            raise ValueError("Unsupported answer mode")
        if mode == "name":
            instruction = (
                "Explain the giver and recipient, then answer with the person's name."
            )
        else:
            instruction = (
                "Explain the giver and recipient, then answer with A or B."
            )
        return (
            f"Story: {self.story}\n"
            f"Choices: A) {self.option_a}  B) {self.option_b}\n"
            f"Question: {self.question}\n"
            f"Instruction: {instruction}\n"
            "Reasoning:"
        )

    def target(self, mode: str) -> str:
        if mode not in ("name", "letter"):
            raise ValueError("Unsupported answer mode")
        rationale = (
            f"Giver: {self.giver}; Recipient: {self.recipient}; "
            "The giver starts the transfer; the recipient gets the item; "
            "Answer: "
        )
        return rationale + (self.name_answer if mode == "name" else self.letter_answer)


def _choice_label(answer: str, option_a: str, option_b: str) -> str:
    if answer == option_a:
        return "A"
    if answer == option_b:
        return "B"
    raise ValueError("Gold answer missing from the two candidate choices")


def make_train(seed: int, rows: int = 3200) -> list[Case]:
    if rows < 40 or rows > 4000 or rows % 4:
        raise ValueError("Training rows must be 40..4000 and divisible by four")
    import itertools
    rng = random.Random(seed)
    configs = [
        (left, right, item, template_index)
        for left, right in itertools.combinations(TRAIN_PEOPLE, 2)
        for item in TRAIN_OBJECTS
        for template_index in range(len(TRAIN_STORIES))
    ]
    episodes = rows // 4
    if episodes > len(configs):
        raise ValueError("Training episode count exceeds unique configurations")
    rng.shuffle(configs)
    cases: list[Case] = []
    seen: set[str] = set()
    for i, (left, right, item, template_index) in enumerate(configs[:episodes]):
        if rng.randrange(2):
            left, right = right, left
        template = TRAIN_STORIES[template_index]
        for direction, (giver, recipient) in enumerate(
                ((left, right), (right, left))):
            story = template.format(giver=giver, recipient=recipient, item=item)
            for role_index, skill in enumerate(SKILLS):
                question_templates = TRAIN_QUESTIONS[skill]
                qtemplate = question_templates[(i + direction) % len(question_templates)]
                question = qtemplate.format(item=item)
                swap_options = (i + direction + role_index) % 2 == 1
                option_a, option_b = (
                    (recipient, giver) if swap_options else (giver, recipient)
                )
                answer = giver if skill == "giver" else recipient
                letter = _choice_label(answer, option_a, option_b)
                case = Case(
                    f"train:{seed}:{i:04d}:{direction}:{skill}",
                    f"train:{seed}:{i:04d}", "training", skill,
                    giver, recipient, item, story, question,
                    option_a, option_b, answer, letter)
                prompt = case.prompt("name")
                if prompt in seen:
                    raise ValueError("Duplicate role-training prompt")
                seen.add(prompt)
                cases.append(case)
    if len(cases) != rows:
        raise RuntimeError("Training case count mismatch")
    validate_balance(cases)
    return cases


def make_final(seed: int = 82001, episodes: int = 60) -> list[Case]:
    if episodes < 20 or episodes > 200 or episodes % 2:
        raise ValueError("Final episode count must be even and 20..200")
    import itertools
    rng = random.Random(seed)
    pairs = tuple(itertools.combinations(FINAL_PEOPLE, 2))
    known = [
        (left, right, item, index % len(FINAL_SEEN_VERB_STORIES))
        for left, right in pairs for item in FINAL_OBJECTS
        for index in range(len(FINAL_SEEN_VERB_STORIES))
    ]
    novel = [
        (left, right, item, index % len(FINAL_UNSEEN_VERB_STORIES))
        for left, right in pairs for item in FINAL_OBJECTS
        for index in range(len(FINAL_UNSEEN_VERB_STORIES))
    ]
    per_slice = episodes // 2
    rng.shuffle(known)
    rng.shuffle(novel)
    selected = [
        ("seen_verb_new_construction", *cfg)
        for cfg in known[:per_slice]
    ] + [
        ("unseen_verb", *cfg)
        for cfg in novel[:per_slice]
    ]
    rng.shuffle(selected)
    cases: list[Case] = []
    seen: set[str] = set()
    for i, (stratum, left, right, item, template_index) in enumerate(selected):
        templates = (FINAL_UNSEEN_VERB_STORIES if stratum == "unseen_verb"
                     else FINAL_SEEN_VERB_STORIES)
        template = templates[template_index]
        for direction, (giver, recipient) in enumerate(
                ((left, right), (right, left))):
            story = template.format(giver=giver, recipient=recipient, item=item)
            for role_index, skill in enumerate(SKILLS):
                qtemplates = FINAL_QUESTIONS[skill]
                question = qtemplates[(i + direction + role_index) % len(qtemplates)]
                question = question.format(item=item)
                swap_options = (i + direction + role_index) % 2 == 1
                option_a, option_b = (
                    (recipient, giver) if swap_options else (giver, recipient)
                )
                answer = giver if skill == "giver" else recipient
                letter = _choice_label(answer, option_a, option_b)
                case = Case(
                    f"final:{seed}:{i:04d}:{direction}:{skill}",
                    f"final:{seed}:{i:04d}", stratum, skill,
                    giver, recipient, item, story, question,
                    option_a, option_b, answer, letter)
                prompt = case.prompt("name")
                if prompt in seen:
                    raise ValueError("Duplicate final prompt")
                seen.add(prompt)
                cases.append(case)
    if len(cases) != episodes * 4:
        raise RuntimeError("Final case count mismatch")
    validate_balance(cases)
    validate_split(make_train(101, 160), cases)
    return cases


def validate_balance(cases: list[Case]) -> None:
    if not cases or any(row.skill not in SKILLS for row in cases):
        raise ValueError("Invalid role labels")
    if any(sum(row.skill == skill for row in cases) * 2 != len(cases)
           for skill in SKILLS):
        raise ValueError("Role classes are not balanced")
    for skill in SKILLS:
        letters = [row.letter_answer for row in cases if row.skill == skill]
        if letters.count("A") != letters.count("B"):
            raise ValueError("Candidate answer letters are not balanced")


def validate_split(train: list[Case], final: list[Case]) -> None:
    if set(TRAIN_PEOPLE) & set(FINAL_PEOPLE):
        raise ValueError("Training and final names overlap")
    if set(TRAIN_OBJECTS) & set(FINAL_OBJECTS):
        raise ValueError("Training and final objects overlap")
    train_prompts = {row.prompt("name") for row in train}
    final_prompts = {row.prompt("name") for row in final}
    if train_prompts & final_prompts:
        raise ValueError("Training and final prompts overlap")
    train_templates = set(TRAIN_STORIES) | {
        q for values in TRAIN_QUESTIONS.values() for q in values
    }
    final_templates = (set(FINAL_SEEN_VERB_STORIES) |
                       set(FINAL_UNSEEN_VERB_STORIES) |
                       {q for values in FINAL_QUESTIONS.values() for q in values})
    if train_templates & final_templates:
        raise ValueError("Final templates overlap training")


def shuffle_targets(cases: list[Case], seed: int) -> list[str]:
    rng = random.Random(seed)
    targets = [row.target("name") for row in cases]
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(cases):
        groups[row.skill].append(i)
    output = targets[:]
    for skill, indexes in groups.items():
        gold = [targets[i] for i in indexes]
        shuffled = gold[:]
        for _ in range(40):
            rng.shuffle(shuffled)
            if shuffled != gold:
                break
        if shuffled == gold:
            raise ValueError(f"Failed to shuffle structured targets for {skill}")
        for index, target in zip(indexes, shuffled):
            output[index] = target
    return output


def _structured_rows(cases: list[Case], mode: str) -> list[DirectExample]:
    return [
        DirectExample(row.id, row.skill, row.prompt(mode), row.target(mode))
        for row in cases
    ]


def _encode_cases(cases: list[Case], mode: str, codec, context: int):
    return encode(_structured_rows(cases, mode), codec, context)


def _episode_indexes(cases: list[Case]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(cases):
        groups[row.episode].append(index)
    if any(len(indices) != 4 for indices in groups.values()):
        raise ValueError("Every event must have four role-reversal questions")
    return dict(groups)


def _parse_output(text: str) -> dict:
    pattern = re.compile(
        r"Giver\s*:\s*([^;]+?)\s*;\s*Recipient\s*:\s*([^;]+?)\s*;"
        r".*?Answer\s*:\s*([A-Za-z]+)\s*\.?",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        return {"giver": "", "recipient": "", "answer": "", "parse_ok": False}
    return {
        "giver": match.group(1).strip().casefold(),
        "recipient": match.group(2).strip().casefold(),
        "answer": match.group(3).strip().casefold(),
        "parse_ok": True,
    }


@torch.no_grad()
def evaluate(model: ByteTransformer, codec, cases: list[Case], mode: str,
             max_tokens: int = 48) -> dict:
    model.eval()
    predictions = []
    for row in cases:
        prompt = codec.encode(row.prompt(mode)).ids
        ids = prompt[:]
        continuation = []
        for _ in range(max_tokens):
            logits, _ = model(torch.tensor(
                [ids[-model.cfg.max_seq_len:]], dtype=torch.long))
            next_id = int(logits[0, -1].argmax())
            ids.append(next_id)
            continuation.append(next_id)
            if "." in codec.decode(continuation):
                break
        decoded = codec.decode(continuation)
        parsed = _parse_output(decoded)
        answer = row.name_answer if mode == "name" else row.letter_answer
        predicted_answer = parsed["answer"]
        predicted_giver = parsed["giver"]
        predicted_recipient = parsed["recipient"]
        predictions.append({
            "id": row.id, "episode": row.episode, "skill": row.skill,
            "stratum": row.stratum, "mode": mode,
            "prompt": row.prompt(mode), "expected_giver": row.giver,
            "expected_recipient": row.recipient, "expected_answer": answer,
            "decoded": decoded,
            "predicted_giver": predicted_giver,
            "predicted_recipient": predicted_recipient,
            "predicted_answer": predicted_answer,
            "parse_ok": parsed["parse_ok"],
            "giver_correct": predicted_giver == row.giver.casefold(),
            "recipient_correct": predicted_recipient == row.recipient.casefold(),
            "answer_correct": predicted_answer == answer.casefold(),
        })

    def summary(rows):
        if not rows:
            return {"correct": 0, "total": 0, "accuracy": 0.0}
        correct = sum(bool(row["answer_correct"]) for row in rows)
        return {"correct": correct, "total": len(rows),
                "accuracy": round(correct / len(rows), 4)}

    episodes: dict[str, list[dict]] = defaultdict(list)
    for prediction in predictions:
        episodes[prediction["episode"]].append(prediction)
    return {
        **summary(predictions),
        "role_trace_both_correct": {
            "correct": sum(
                all(row["giver_correct"] and row["recipient_correct"] for row in rows)
                for rows in episodes.values()),
            "total": len(episodes),
        },
        "giver_trace_accuracy": round(
            sum(x["giver_correct"] for x in predictions) / len(predictions), 4),
        "recipient_trace_accuracy": round(
            sum(x["recipient_correct"] for x in predictions) / len(predictions), 4),
        "by_skill": {
            skill: summary([row for row in predictions if row["skill"] == skill])
            for skill in SKILLS
        },
        "by_slice": {
            name: summary([row for row in predictions if row["stratum"] == name])
            for name in sorted({row["stratum"] for row in predictions})
        },
        "episodes_all_four_answers_correct": {
            "correct": sum(all(row["answer_correct"] for row in rows)
                           for rows in episodes.values()),
            "total": len(episodes),
        },
        "predictions": predictions,
        "scoring": "Greedy structured rationale and exact final answer",
    }


def _loss(model, valid_tokens) -> float:
    return round(heldout(model, valid_tokens, windows=64, seq_len=128), 6)


def run(*, folder: Path, out: Path, source_run: int = 42085634121,
        seeds: tuple[int, ...] = (101, 202, 303),
        stages=STAGES) -> dict:
    if (not 2 <= len(seeds) <= 4 or len(set(seeds)) != len(seeds)
        or any(seed < 1 or seed > 1_000_000 for seed in seeds)):
        raise ValueError("Use 2..4 distinct bounded seeds")
    if (not stages or len(stages) > 4
        or any(count < 40 or count > 4000 or count % 4 or steps < 1
               or steps > 2500 or not 0 <= weight < 1 or not 0 < lr < 0.01
               for count, steps, weight, lr in stages)
        or sum(stage[1] for stage in stages) > 4000):
        raise ValueError("Curriculum exceeds bounded CPU run limits")
    if any(stages[i][0] > stages[i + 1][0] for i in range(len(stages)-1)):
        raise ValueError("Curriculum must grow monotonically")

    torch.set_num_threads(min(2, torch.get_num_threads()))
    checkpoint, manifest, codec = verify_source(folder)
    source_hash = hashlib.sha256((folder / "checkpoint.pt").read_bytes()).hexdigest()
    public_train = torch.tensor(
        codec.encode((folder / "train.txt").read_text()).ids, dtype=torch.long)
    public_valid = torch.tensor(
        codec.encode((folder / "valid.txt").read_text()).ids, dtype=torch.long)
    final_cases = make_final()
    final_hash = hashlib.sha256("\n".join(
        f"{case.prompt('name')}\t{case.name_answer}\t{case.letter_answer}"
        for case in final_cases).encode()).hexdigest()

    seed_results = []
    start_all = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        train_cases = make_train(seed, stages[-1][0])
        validate_split(train_cases, final_cases)
        train_hash = hashlib.sha256("\n".join(
            f"{case.prompt('name')}\t{case.target('name')}"
            for case in train_cases).encode()).hexdigest()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            base = ByteTransformer(ModelConfig(**checkpoint["configuration"]))
            base.load_state_dict(checkpoint["model"], strict=True)
            models = {
                name: __import__("copy").deepcopy(base)
                for name in ARMS
            }
        initial_loss = _loss(base, public_valid)
        encoded_name = _encode_cases(train_cases, "name", codec, base.cfg.max_seq_len)
        encoded_choice = _encode_cases(train_cases, "letter", codec, base.cfg.max_seq_len)
        null_targets = shuffle_targets(train_cases, seed + 197)
        encoded_null = encode([
            DirectExample(case.id, case.skill, case.prompt("name"), target)
            for case, target in zip(train_cases, null_targets)
        ], codec, base.cfg.max_seq_len)
        group_full = _episode_indexes(train_cases)
        stage_history = []

        for stage_index, (count, steps, text_weight, lr) in enumerate(stages, 1):
            rows = train_cases[:count]
            group = _episode_indexes(rows)
            if set(group) - set(group_full):
                raise ValueError("Curriculum stage escaped the pinned training split")
            rng = random.Random(seed + stage_index * 313)
            keys = tuple(sorted(group))
            schedule = [rng.choice(keys) for _ in range(steps)]
            name_stage = encoded_name[:count]
            choice_stage = encoded_choice[:count]
            null_stage = encoded_null[:count]

            def adapt(model, arm, encoded=None):
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
                language_rng = torch.Generator().manual_seed(seed + stage_index * 1109)
                model.train()
                last = None
                for episode in schedule:
                    if arm == "text_only":
                        # Two language forwards per update approximate the QA+
                        # rehearsal compute of the supervised arms.
                        x1, y1 = make_batch(public_train, 4, 52, language_rng)
                        x2, y2 = make_batch(public_train, 4, 52, language_rng)
                        _, loss1 = model(x1, y1)
                        _, loss2 = model(x2, y2)
                        total = (loss1 + loss2) / 2
                        last = round(float(total.detach()), 6)
                    else:
                        indexes = group[episode]
                        task_loss = answer_loss(model, [encoded[i] for i in indexes])
                        x, y = make_batch(public_train, 2, 64, language_rng)
                        _, lm_loss = model(x, y)
                        total = (1 - text_weight) * task_loss + text_weight * lm_loss
                        last = round(float(task_loss.detach()), 6)
                    optimizer.zero_grad(set_to_none=True)
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                return last

            name_loss = adapt(models["structured_names"], "structured_names",
                              name_stage)
            choice_loss = adapt(models["structured_choice"], "structured_choice",
                                choice_stage)
            null_loss = adapt(models["shuffled_labels"], "shuffled_labels",
                               null_stage)
            text_loss = adapt(models["text_only"], "text_only")
            stage_history.append({
                "stage": stage_index, "training_rows": count,
                "counterfactual_episodes": count // 4,
                "updates_per_arm": steps, "language_rehearsal_weight": text_weight,
                "learning_rate": lr,
                "structured_name_last_task_loss": name_loss,
                "structured_choice_last_task_loss": choice_loss,
                "shuffled_last_task_loss": null_loss,
                "text_only_last_loss": text_loss,
                "text_only_compute_proxy": "two 4x52 LM batches per update",
                "supervised_compute_proxy": "one four-example QA batch plus one 2x64 LM batch",
            })
            print(json.dumps({"seed": seed, "stage": stage_index,
                              "updates_per_arm": steps}), flush=True)

        # Blind final evaluation occurs after training; no checkpoint selection
        # or schedule change uses these scores.
        base_name = evaluate(base, codec, final_cases, "name")
        gold_name = evaluate(models["structured_names"], codec, final_cases, "name")
        gold_choice = evaluate(models["structured_choice"], codec, final_cases, "letter")
        null_name = evaluate(models["shuffled_labels"], codec, final_cases, "name")
        text_name = evaluate(models["text_only"], codec, final_cases, "name")
        final_loss = {arm: _loss(model, public_valid)
                      for arm, model in models.items()}
        seed_result = {
            "seed": seed,
            "source_training_run": source_run,
            "source_checkpoint_sha256": source_hash,
            "source_manifest_train_sha256": manifest["train_sha256"],
            "source_step": checkpoint["step"],
            "source_parameters": base.num_parameters,
            "train_rows": len(train_cases),
            "train_episodes": len(train_cases) // 4,
            "train_sha256": train_hash,
            "final_rows": len(final_cases),
            "final_episodes": len(final_cases) // 4,
            "final_shared_sha256": final_hash,
            "baseline_name": base_name,
            "structured_name": gold_name,
            "structured_choice": gold_choice,
            "shuffled_name_control": null_name,
            "equal_compute_text_only": text_name,
            "language_loss_before": initial_loss,
            "language_loss_after": final_loss,
            "stages": stage_history,
        }
        seed_results.append(seed_result)
        for arm in ("structured_names", "structured_choice",
                    "shuffled_labels", "text_only"):
            torch.save({
                "schema": 6, "experimental_only": True, "arm": arm,
                "seed": seed, "source_training_run": source_run,
                "source_checkpoint_sha256": source_hash,
                "train_sha256": train_hash, "final_shared_sha256": final_hash,
                "configuration": asdict(models[arm].cfg),
                "model": models[arm].state_dict(),
            }, out / f"{arm}-seed-{seed}.pt")
        del models, base
        print(json.dumps({
            "seed": seed,
            "baseline_name": base_name["correct"],
            "structured_name": gold_name["correct"],
            "structured_choice": gold_choice["correct"],
            "shuffled": null_name["correct"],
            "text_only": text_name["correct"],
            "role_trace": gold_choice["role_trace_both_correct"],
            "language_loss_before": initial_loss,
            "language_loss_name": final_loss["structured_names"],
        }), flush=True)

    # Aggregate raw correct/total over the independent training seeds.
    def aggregate(key):
        correct = sum(r[key]["correct"] for r in seed_results)
        total = sum(r[key]["total"] for r in seed_results)
        return {"correct": correct, "total": total,
                "accuracy": round(correct / total, 4) if total else 0.0}
    report = {
        "schema": 1, "status": "research_only_not_promoted",
        "source_training_run": source_run,
        "source_checkpoint_sha256": source_hash,
        "source_manifest_train_sha256": manifest["train_sha256"],
        "seeds": list(seeds),
        "final_shared_across_seeds": True,
        "final_source": "internally authored deterministic program labels",
        "training_rows_per_seed": stages[-1][0],
        "training_dataset_hashes": [r["train_sha256"] for r in seed_results],
        "final_rows": len(final_cases), "final_episodes": len(final_cases)//4,
        "final_sha256": final_hash,
        "disjoint_names_objects_templates": True,
        "final_seen_verb_new_construction_rows": sum(
            row.stratum == "seen_verb_new_construction" for row in final_cases),
        "final_unseen_verb_rows": sum(
            row.stratum == "unseen_verb" for row in final_cases),
        "methods": [
            "structured role explanation + direct name",
            "structured role explanation + candidate letter",
            "structured target with within-role shuffled labels",
            "equal-compute text-only continuation",
        ],
        "same_optimizer_updates_and_source_checkpoint": True,
        "results_by_seed": seed_results,
        "aggregate": {
            "frozen_name": aggregate("baseline_name"),
            "structured_name": aggregate("structured_name"),
            "structured_choice": aggregate("structured_choice"),
            "shuffled_name": aggregate("shuffled_name_control"),
            "text_only_name": aggregate("equal_compute_text_only"),
        },
        "runtime_seconds": round(time.monotonic() - start_all, 2),
        "no_external_ai_teacher_was_run": True,
        "no_private_data": True,
        "checkpoint_promoted": False,
        "limitations": [
            "Internally authored synthetic transfer benchmark; no external natural-language labels",
            "Final challenge families are fixed across seeds; training examples and initialization vary",
            "Exact generated names and role traces remain sensitive to tokenizer spelling",
            "Candidate-letter accuracy isolates role selection only within this task format",
            "A structured explanation is supervised text, not a verified account of hidden internal reasoning",
            "Requires later evaluation on independently sourced licensed role-labeled text",
        ],
    }
    (out / "role-reasoning-v2-results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "aggregate": report["aggregate"],
        "runtime_seconds": report["runtime_seconds"],
        "final_rows_per_seed": report["final_rows"],
    }, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/role-reasoning-v2"))
    parser.add_argument("--source-run", type=int, default=42085634121)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    args = parser.parse_args()
    run(folder=args.checkpoint_dir, out=args.out,
        source_run=args.source_run, seeds=tuple(args.seeds))


if __name__ == "__main__":
    main()
