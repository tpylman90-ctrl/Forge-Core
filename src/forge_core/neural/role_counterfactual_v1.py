"""Focused counterfactual giver/recipient lesson for Forge Seed v0.7.

Each training episode contains two minimal-pair stories with the direction
of transfer reversed. For each story Forge must generate both the giver and
recipient names. Final tests use disjoint names and objects, unseen sentence
and question templates, and include a separately reported unseen-verb slice.

The candidate is trained from the frozen scratch-origin language checkpoint
with matched language rehearsal; a within-role shuffled-answer model controls
for optimization and answer frequencies. Final questions are scored once
after the fixed training schedule. Research only; no promotion.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import re
import time

import torch

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.preschool_direct import (
    DirectExample, answer_loss, encode, normalize_answer,
    verify_source,
)
from forge_core.neural.seed_v07 import heldout, generate_samples
from forge_core.neural.train import make_batch


TRAIN_PEOPLE = ("Nora", "Owen", "Ella", "Finn", "Sara", "June",
                "Liam", "Arlo", "Zoe", "Hugo", "Ruby", "Will")
FINAL_PEOPLE = ("Mila", "Iris", "Oscar", "Evan", "Lila", "Milo",
                "Cora", "Noah", "Rose", "Theo", "Lena", "Jude")
TRAIN_OBJECTS = ("scarf", "hat", "kite", "book", "toy", "cup", "ball", "coat")
FINAL_OBJECTS = ("backpack", "basket", "ribbon", "blanket",
                 "flower", "pillow", "pencil", "wagon")

TRAIN_STORY_TEMPLATES = (
    "{giver} gave the {item} to {recipient}.",
    "{giver} handed the {item} to {recipient}.",
    "{giver} passed the {item} over to {recipient}.",
    "{recipient} received the {item} from {giver}.",
)
FINAL_SEEN_VERB_STORIES = (
    "{giver} gave {recipient} the {item}.",
    "{recipient} got the {item} from {giver}.",
)
FINAL_UNSEEN_VERB_STORIES = (
    "{giver} transferred the {item} into {recipient}'s care.",
    "The {item} was delivered to {recipient} by {giver}.",
)
TRAIN_QUESTIONS = {
    "giver": "Who gave the {item}?",
    "recipient": "Who received the {item}?",
}
FINAL_QUESTIONS = {
    "giver": (
        "Who handed over the {item}?",
        "From whom did the {item} come?",
    ),
    "recipient": (
        "Who ended up with the {item}?",
        "To whom was the {item} delivered?",
    ),
}
SKILLS = ("giver", "recipient")


@dataclass(frozen=True)
class RoleExample:
    id: str
    episode: str
    split: str
    stratum: str
    skill: str
    prompt: str
    answer: str

    def direct(self) -> DirectExample:
        return DirectExample(self.id, self.skill, self.prompt, self.answer)


def _story(template: str, giver: str, recipient: str, item: str) -> str:
    return template.format(giver=giver, recipient=recipient, item=item)


def make_train(count: int, seed: int = 1811) -> list[RoleExample]:
    """Return count short-answer rows, grouped into four-row counterfactual episodes."""
    if count < 40 or count > 4000 or count % 4:
        raise ValueError("Training rows must be 40..4000 and divisible by four")
    rng = random.Random(seed)
    rows: list[RoleExample] = []
    seen: set[str] = set()
    episodes = count // 4
    for i in range(episodes):
        left, right = rng.sample(TRAIN_PEOPLE, 2)
        item = rng.choice(TRAIN_OBJECTS)
        template = TRAIN_STORY_TEMPLATES[i % len(TRAIN_STORY_TEMPLATES)]
        for direction, (giver, recipient) in enumerate(((left, right), (right, left))):
            story = _story(template, giver, recipient, item)
            for skill in SKILLS:
                question = TRAIN_QUESTIONS[skill].format(item=item)
                prompt = f"Story: {story}\nQuestion: {question}\nAnswer:"
                if prompt in seen:
                    raise ValueError("Duplicate training prompt")
                seen.add(prompt)
                answer = giver if skill == "giver" else recipient
                rows.append(RoleExample(
                    f"train:{i:04d}:{direction}:{skill}", f"train:{i:04d}",
                    "training", skill, skill, prompt, answer))
    if len(rows) != count:
        raise RuntimeError("Training generator row count mismatch")
    _validate_balance(rows)
    return rows


def make_final(episodes: int = 50, seed: int = 9187) -> list[RoleExample]:
    """Untouched final set: separate entities, objects, wording, and verb slice.

    Half the episodes use training-seen give/get verbs in new constructions.
    The other half use transfer/deliver, verbs absent from every training item.
    """
    if episodes < 20 or episodes > 200 or episodes % 2:
        raise ValueError("Final episode count must be even and in 20..200")
    rng = random.Random(seed)
    rows: list[RoleExample] = []
    seen: set[str] = set()
    for i in range(episodes):
        left, right = rng.sample(FINAL_PEOPLE, 2)
        item = rng.choice(FINAL_OBJECTS)
        unseen_verb = i >= episodes // 2
        stratum = "unseen_verb" if unseen_verb else "seen_verb_new_construction"
        templates = FINAL_UNSEEN_VERB_STORIES if unseen_verb else FINAL_SEEN_VERB_STORIES
        template = templates[i % len(templates)]
        for direction, (giver, recipient) in enumerate(((left, right), (right, left))):
            story = _story(template, giver, recipient, item)
            for role_index, skill in enumerate(SKILLS):
                question_templates = FINAL_QUESTIONS[skill]
                question = question_templates[(i + direction + role_index) % len(question_templates)]
                prompt = f"Story: {story}\nQuestion: {question.format(item=item)}\nAnswer:"
                if prompt in seen:
                    raise ValueError("Duplicate final prompt")
                seen.add(prompt)
                answer = giver if skill == "giver" else recipient
                rows.append(RoleExample(
                    f"final:{i:04d}:{direction}:{skill}", f"final:{i:04d}",
                    "final", stratum, skill, prompt, answer))
    if len(rows) != episodes * 4:
        raise RuntimeError("Final generator row count mismatch")
    _validate_balance(rows)
    _validate_splits(make_train(160, seed + 1), rows)
    return rows


def _validate_balance(rows: list[RoleExample]) -> None:
    if not rows or any(row.skill not in SKILLS for row in rows):
        raise ValueError("Missing role-label examples")
    if any(sum(row.skill == skill for row in rows) * 2 != len(rows)
           for skill in SKILLS):
        raise ValueError("Giver and recipient rows must be exactly balanced")
    if len({row.prompt for row in rows}) != len(rows):
        raise ValueError("Prompts must be unique within each split")


def _validate_splits(train: list[RoleExample], final: list[RoleExample]) -> None:
    if set(TRAIN_PEOPLE) & set(FINAL_PEOPLE):
        raise ValueError("Final people overlap training vocabulary")
    if set(TRAIN_OBJECTS) & set(FINAL_OBJECTS):
        raise ValueError("Final objects overlap training vocabulary")
    if {row.prompt for row in train} & {row.prompt for row in final}:
        raise ValueError("Training/final prompt overlap")
    train_text = "\n".join(row.prompt for row in train).casefold()
    final_text = "\n".join(row.prompt for row in final).casefold()
    if any(verb in train_text for verb in ("transferred", "delivered")):
        raise ValueError("Held-out verb leaked into training")
    if not all(verb in final_text for verb in ("transferred", "delivered")):
        raise ValueError("Final unseen-verb slice is incomplete")


def shuffle_roles(rows: list[RoleExample], seed: int) -> list[RoleExample]:
    """Shuffle names within giver/recipient classes without changing prompts."""
    rng = random.Random(seed)
    pools = {skill: [row.answer for row in rows if row.skill == skill]
             for skill in SKILLS}
    for skill, values in pools.items():
        original = values[:]
        for _ in range(30):
            rng.shuffle(values)
            if values != original:
                break
        if values == original:
            raise ValueError(f"Cannot create a shuffled-label control for {skill}")
    cursor = {skill: 0 for skill in SKILLS}
    shuffled = []
    for row in rows:
        idx = cursor[row.skill]
        cursor[row.skill] += 1
        shuffled.append(RoleExample(
            row.id, row.episode, row.split, row.stratum,
            row.skill, row.prompt, pools[row.skill][idx]))
    if [row.prompt for row in rows] != [row.prompt for row in shuffled]:
        raise AssertionError("Shuffled arm prompts differ")
    return shuffled


@torch.no_grad()
def evaluate(model: ByteTransformer, codec, rows: list[RoleExample],
             max_tokens: int = 8) -> dict:
    model.eval()
    predictions = []
    for row in rows:
        prefix = codec.encode(row.prompt).ids
        ids = prefix[:]
        continuation = []
        for _ in range(max_tokens):
            logits, _ = model(torch.tensor(
                [ids[-model.cfg.max_seq_len:]], dtype=torch.long))
            nxt = int(logits[0, -1].argmax())
            ids.append(nxt)
            continuation.append(nxt)
            if "." in codec.decode(continuation):
                break
        predicted = codec.decode(continuation)
        predictions.append({
            "id": row.id, "episode": row.episode, "stratum": row.stratum,
            "skill": row.skill, "prompt": row.prompt, "expected": row.answer,
            "predicted": predicted,
            "correct": normalize_answer(predicted) == normalize_answer(row.answer),
        })
    def summarize(items):
        correct = sum(bool(item["correct"]) for item in items)
        total = len(items)
        return {"correct": correct, "total": total,
                "accuracy": round(correct / total, 4) if total else 0.0}
    episode_rows: dict[str, list[dict]] = {}
    for item in predictions:
        episode_rows.setdefault(item["episode"], []).append(item)
    return {
        **summarize(predictions),
        "by_role": {skill: summarize([p for p in predictions if p["skill"] == skill])
                    for skill in SKILLS},
        "by_slice": {
            name: summarize([p for p in predictions if p["stratum"] == name])
            for name in sorted({p["stratum"] for p in predictions})
        },
        "episodes_all_four_correct": summarize([
            {"correct": all(item["correct"] for item in episode)}
            for episode in episode_rows.values()
        ]),
        "predictions": predictions,
        "scoring": "Greedy generated short answer; normalized exact name match",
    }


def _indices_by_episode(rows: list[RoleExample]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        groups.setdefault(row.episode, []).append(i)
    if any(len(group) != 4 for group in groups.values()):
        raise ValueError("Every teaching episode must contain four paired examples")
    return groups


def run(*, folder: Path, out: Path, source_run: int = 42085634121,
        seed: int = 1811,
        stages: tuple[tuple[int, int, float, float], ...] = (
            (160, 250, 0.05, 2e-4),
            (800, 650, 0.15, 1e-4),
            (3200, 1600, 0.25, 6e-5),
        )) -> dict:
    if (not 1 <= len(stages) <= 4 or
        any(count < 40 or count > 4000 or count % 4 or steps < 1
            or steps > 2500 or not 0 <= weight < 1
            or not 0 < lr < 0.01 for count, steps, weight, lr in stages)
        or sum(stage[1] for stage in stages) > 4000):
        raise ValueError("Counterfactual curriculum outside bounded limits")
    if any(stages[i][0] > stages[i + 1][0] for i in range(len(stages) - 1)):
        raise ValueError("Curriculum stages must grow monotonically")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    ckpt, manifest, codec = verify_source(folder)
    original_hash = hashlib.sha256((folder / "checkpoint.pt").read_bytes()).hexdigest()
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        frozen = ByteTransformer(ModelConfig(**ckpt["configuration"]))
        frozen.load_state_dict(ckpt["model"])
        verified, shuffled = __import__("copy").deepcopy(frozen), __import__("copy").deepcopy(frozen)

    train_full = make_train(stages[-1][0], seed)
    final = make_final(50, seed + 7001)
    _validate_splits(train_full, final)
    train_sha = hashlib.sha256("\n".join(
        f"{row.prompt}\t{row.answer}" for row in train_full).encode()).hexdigest()
    final_sha = hashlib.sha256("\n".join(
        f"{row.prompt}\t{row.answer}" for row in final).encode()).hexdigest()
    all_train_direct = [row.direct() for row in train_full]
    full_encoded = encode(all_train_direct, codec, frozen.cfg.max_seq_len)
    public_train = torch.tensor(
        codec.encode((folder / "train.txt").read_text()).ids, dtype=torch.long)
    public_valid = torch.tensor(
        codec.encode((folder / "valid.txt").read_text()).ids, dtype=torch.long)

    def language_loss(model):
        return round(heldout(model, public_valid, windows=64, seq_len=128), 6)

    initial_language_loss = language_loss(frozen)
    history = []
    start = time.monotonic()
    for ordinal, (count, steps, text_weight, lr) in enumerate(stages, 1):
        rows = train_full[:count]
        null_rows = shuffle_roles(rows, seed + ordinal * 313)
        gold_encoded = full_encoded[:count]
        null_encoded = encode([row.direct() for row in null_rows],
                              codec, frozen.cfg.max_seq_len)
        episode_groups = _indices_by_episode(rows)
        if _indices_by_episode(null_rows) != episode_groups:
            raise AssertionError("Counterfactual schedule differs across arms")
        rng = random.Random(seed + ordinal * 811)
        episode_keys = tuple(sorted(episode_groups))
        schedule = [rng.choice(episode_keys) for _ in range(steps)]

        def adapt(model, encoded, arm):
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            text_rng = torch.Generator().manual_seed(seed + ordinal * 1297)
            model.train()
            last_loss = None
            for step, episode in enumerate(schedule, 1):
                indexes = episode_groups[episode]
                task_loss = answer_loss(model, [encoded[i] for i in indexes])
                if text_weight:
                    x, y = make_batch(public_train, 2, 64, text_rng)
                    _, lm_loss = model(x, y)
                    total_loss = (1 - text_weight) * task_loss + text_weight * lm_loss
                else:
                    total_loss = task_loss
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                last_loss = round(float(task_loss.detach()), 6)
            return last_loss

        gold_loss = adapt(verified, gold_encoded, "verified")
        null_loss = adapt(shuffled, null_encoded, "shuffled")
        history.append({
            "stage": ordinal, "paired_episodes": count // 4,
            "direct_answer_rows": count, "updates_per_arm": steps,
            "language_rehearsal_weight": text_weight,
            "learning_rate": lr, "verified_final_batch_loss": gold_loss,
            "shuffled_final_batch_loss": null_loss,
            "verified_language_loss": language_loss(verified),
            "shuffled_language_loss": language_loss(shuffled),
        })
        print(json.dumps({
            "stage": ordinal, "episodes": count // 4, "updates": steps,
            "verified_language_loss": history[-1]["verified_language_loss"],
        }), flush=True)

    # The final split is held untouched until this point. Score all arms once.
    baseline = evaluate(frozen, codec, final)
    gold_result = evaluate(verified, codec, final)
    null_result = evaluate(shuffled, codec, final)
    end_language_loss = language_loss(verified)
    report = {
        "schema": 1, "status": "research_only_not_promoted",
        "source_training_run": source_run, "seed": seed,
        "source_checkpoint_sha256": original_hash,
        "source_manifest_train_sha256": manifest["train_sha256"],
        "source_training_steps": ckpt["step"],
        "source_model_parameters": frozen.num_parameters,
        "curriculum": [asdict_stage(stage) for stage in stages],
        "training_paired_episodes": stages[-1][0] // 4,
        "final_episodes": 50, "final_answer_rows": len(final),
        "train_names": list(TRAIN_PEOPLE), "final_names": list(FINAL_PEOPLE),
        "train_objects": list(TRAIN_OBJECTS), "final_objects": list(FINAL_OBJECTS),
        "final_split_has_disjoint_names_objects_templates": True,
        "train_sha256": train_sha, "final_sha256": final_sha,
        "train_rows": len(train_full), "final_unseen_verb_rows": sum(
            row.stratum == "unseen_verb" for row in final),
        "frozen_final": baseline, "verified_final": gold_result,
        "shuffled_control_final": null_result,
        "language_loss_before": initial_language_loss,
        "language_loss_after_verified": end_language_loss,
        "language_loss_retained_within_five_percent":
            end_language_loss <= initial_language_loss * 1.05,
        "stages": history,
        "fixed_story_samples": generate_samples(verified, codec, seed, new_tokens=64),
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "limitations": [
            "Internally authored synthetic transfer facts, not natural conversation",
            "Final names, objects and sentence/question constructions are disjoint from training",
            "Half of final cases also use verbs absent from training",
            "Exact name match may penalize correct paraphrases",
            "One seed; results require fresh data and multi-seed replication",
            "A task-specific synthetic result does not establish general comprehension",
            "No automatic weight promotion or Android integration",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "role-counterfactual-results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for arm, model in (("verified", verified), ("shuffled", shuffled)):
        torch.save({
            "schema": 5, "experimental_only": True, "arm": arm,
            "method": "paired_giver_recipient_counterfactual_v1",
            "source_training_run": source_run,
            "source_checkpoint_sha256": original_hash,
            "configuration": asdict(model.cfg), "model": model.state_dict(),
        }, out / f"{arm}-research.pt")
    print(json.dumps({
        "source_training_run": source_run,
        "frozen": baseline["correct"], "verified": gold_result["correct"],
        "shuffled": null_result["correct"], "total": len(final),
        "giver": gold_result["by_role"]["giver"],
        "recipient": gold_result["by_role"]["recipient"],
        "seen_verb_new_construction": gold_result["by_slice"]["seen_verb_new_construction"],
        "unseen_verb": gold_result["by_slice"]["unseen_verb"],
        "all_four_per_episode": gold_result["episodes_all_four_correct"],
        "language_loss_before": initial_language_loss,
        "language_loss_after": end_language_loss,
        "seconds": report["elapsed_seconds"],
    }, indent=2), flush=True)
    return report


def asdict_stage(stage):
    return {"training_rows": stage[0], "updates_per_arm": stage[1],
            "language_rehearsal_weight": stage[2], "learning_rate": stage[3]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/role-counterfactual"))
    parser.add_argument("--source-run", type=int, default=42085634121)
    parser.add_argument("--seed", type=int, default=1811)
    args = parser.parse_args()
    run(folder=args.checkpoint_dir, out=args.out,
        source_run=args.source_run, seed=args.seed)


if __name__ == "__main__":
    main()
