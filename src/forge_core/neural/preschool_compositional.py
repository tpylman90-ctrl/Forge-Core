"""Compositional language apprenticeship: generalize from new wording and names.

All examples are Forge Core-authored, deterministically labeled, nonpersonal
synthetic facts. TRAIN and CHALLENGE use disjoint names, objects, and sentence
and question templates. A challenge question is never used as training data.
The research-only original scratch-trained Forge Seed is preserved unchanged.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import torch

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.preschool_direct import (
    DirectExample, answer_loss, encode, evaluate_direct, shuffle_answers,
)
from forge_core.neural.preschool_qa import SKILLS, verify_source
from forge_core.neural.seed_v07 import heldout, generate_samples
from forge_core.neural.train import make_batch


# Note: the roles "giver" and "recipient" are distinct supervised questions.
SKILL_NAMES = ("color", "recipient", "giver", "location", "count")
TRAIN_PEOPLE = ("Nora", "Owen", "Ella", "Finn", "Sara", "June",
                "Liam", "Arlo", "Zoe", "Hugo", "Ruby", "Will")
CHALLENGE_PEOPLE = ("Mila", "Iris", "Oscar", "Evan", "Lila", "Milo",
                    "Cora", "Noah", "Rose", "Theo", "Lena", "Jude")
TRAIN_OBJECTS = ("scarf", "hat", "kite", "book", "toy", "cup", "ball", "coat")
CHALLENGE_OBJECTS = ("backpack", "basket", "ribbon", "blanket",
                     "flower", "pillow", "pencil", "wagon")
COLORS = ("red", "blue", "green", "yellow", "pink", "white",
          "black", "purple", "orange", "brown")
PLACES = ("under", "beside", "behind", "near", "above", "below",
          "inside", "outside")
COUNTS = ("two", "three", "four", "five", "six", "seven",
          "eight", "nine", "ten")


def make_examples(split: str, count: int, seed: int = 707) -> list[DirectExample]:
    """Gold is determined by the selected actor/attribute, never a guessed label.

    Each example is generated independently, with a counterbalanced choice of
    target entity. Challenge-only names, objects, narratives and question
    constructions are absent from the training generator.
    """
    if split not in ("train", "challenge"):
        raise ValueError("Split must be train or challenge")
    if not 40 <= count <= 8000 or count % 10:
        raise ValueError("Example count must be a multiple of 10, from 40 to 8000")
    rng = random.Random(seed + (0 if split == "train" else 700001))
    names = TRAIN_PEOPLE if split == "train" else CHALLENGE_PEOPLE
    objects = TRAIN_OBJECTS if split == "train" else CHALLENGE_OBJECTS
    results: list[DirectExample] = []
    seen = set()

    def fmt(skill: str, a: str, b: str, item: str, item2: str,
            x: str, y: str, target_b: bool, variant: int):
        person = b if target_b else a
        object_ = item2 if target_b else item
        feature = y if target_b else x
        if skill == "color":
            if split == "train":
                stories = (
                    f"{a} carried a {x} {item}. {b} carried a {y} {item2}.",
                    f"{a} brought a {x} {item}, while {b} had a {y} {item2}.",
                    f"The {item} belonging to {a} was {x}. The {item2} belonging to {b} was {y}.",
                    f"{b} showed a {y} {item2}, and {a} showed a {x} {item}.",
                )
                questions = (
                    f"What color was {person}'s {object_}?",
                    f"Name the color of the {object_} that {person} had.",
                    f"Which color did {person} have on the {object_}?",
                )
            else:
                stories = (
                    f"At the market, {a} chose the {x} {item}; {b} selected the {y} {item2}.",
                    f"The tag on {b}'s {item2} said {y}. The tag on {a}'s {item} said {x}.",
                    f"{a}'s {item} looked {x}, but {b}'s {item2} looked {y}.",
                )
                questions = (
                    f"Tell me the color of {person}'s {object_}.",
                    f"Which color described the {object_} of {person}?",
                )
            answer = feature
        elif skill in ("recipient", "giver"):
            if split == "train":
                stories = (
                    f"{a} gave the {item} to {b}.",
                    f"{b} received the {item} from {a}.",
                    f"The {item} passed from {a} to {b}.",
                    f"{a} handed {b} the {item}.",
                )
                questions = (
                    (f"Who got the {item}?", f"Who received the {item}?")
                    if skill == "recipient" else
                    (f"Who gave away the {item}?", f"Who handed over the {item}?")
                )
            else:
                stories = (
                    f"With a smile, {a} passed the {item} into {b}'s hands.",
                    f"After {a} offered the {item}, {b} took it.",
                    f"The {item} was delivered to {b} by {a}.",
                )
                questions = (
                    (f"Who ended up holding the {item}?",
                     f"Who was given the {item}?") if skill == "recipient"
                    else (f"Who handed over the {item} at the start?",
                          f"From whom did the {item} come?")
                )
            answer = b if skill == "recipient" else a
        elif skill == "location":
            if split == "train":
                stories = (
                    f"{a} placed the {item} {x} the box. {b} put the {item2} {y} the box.",
                    f"The {item2} was {y} the box, but the {item} was {x} it.",
                    f"{b} left the {item2} {y} a box. {a} left the {item} {x} the same box.",
                    f"The {item} belonging to {a} sat {x} the box; {b}'s {item2} sat {y} it.",
                )
                questions = (
                    f"Where was the {object_}?",
                    f"Where did {person} put the {object_}?",
                )
            else:
                stories = (
                    f"Next to {a}, the {item} rested {x} a box; the {item2} lay {y} it.",
                    f"The {item2} that {b} brought was {y} the box. {a}'s {item} was {x} the box.",
                    f"{a} looked for a {item}: it was {x} the box. {b}'s {item2} was {y} it.",
                )
                questions = (
                    f"What was the position of the {object_} relative to the box?",
                    f"Find the {object_}: where did it sit?",
                )
            answer = feature
        elif skill == "count":
            if split == "train":
                stories = (
                    f"{a} collected {x} stones. {b} collected {y} shells.",
                    f"{b} found {y} shells, and {a} found {x} stones.",
                    f"The basket of {a} had {x} stones; {b}'s basket had {y} shells.",
                    f"{a} picked up {x} stones. Later {b} picked up {y} shells.",
                )
                questions = (
                    f"How many {'shells' if target_b else 'stones'} did {person} collect?",
                    f"Give the number of {'shells' if target_b else 'stones'} for {person}.",
                )
            else:
                stories = (
                    f"During a walk, {a} gathered {x} stones and {b} found {y} shells.",
                    f"The total in {b}'s pile of shells was {y}; {a}'s pile of stones was {x}.",
                    f"{a} kept {x} stones in a pouch while {b} stored {y} shells.",
                )
                questions = (
                    f"State how many {'shells' if target_b else 'stones'} {person} had.",
                    f"What count of {'shells' if target_b else 'stones'} belonged to {person}?",
                )
            answer = feature
        else:
            raise ValueError("Unsupported teaching skill")
        return stories[variant % len(stories)], questions[
            (variant // len(stories)) % len(questions)], answer

    # Exactly the same number of examples for each skill, target position
    # and family cycle, to limit shortcut label distributions.
    for i in range(count):
        skill = SKILL_NAMES[i % len(SKILL_NAMES)]
        a, b = rng.sample(names, 2)
        item, item2 = rng.sample(objects, 2)
        if skill == "color":
            x, y = rng.sample(COLORS, 2)
        elif skill == "location":
            x, y = rng.sample(PLACES, 2)
        elif skill == "count":
            x, y = rng.sample(COUNTS, 2)
        else:
            x, y = "", ""
        target_b = (i // len(SKILL_NAMES)) % 2 == 1
        # Recipient/giver role is counterbalanced by swapping who is giver.
        if skill in ("recipient", "giver") and target_b:
            a, b = b, a
        variant = i // (len(SKILL_NAMES) * 2)
        story, question, answer = fmt(
            skill, a, b, item, item2, x, y, target_b, variant)
        prompt = f"Story: {story}\nQuestion: {question}\nAnswer:"
        # Very rarely two random draws collide; redraw in a reproducible
        # bounded way without duplicating challenge or training entries.
        attempts = 0
        while prompt in seen and attempts < 20:
            a, b = rng.sample(names, 2)
            item, item2 = rng.sample(objects, 2)
            story, question, answer = fmt(
                skill, a, b, item, item2, x, y, target_b, variant + attempts)
            prompt = f"Story: {story}\nQuestion: {question}\nAnswer:"
            attempts += 1
        if prompt in seen:
            raise ValueError("Synthetic prompt generator exhausted unique examples")
        seen.add(prompt)
        if answer.casefold() not in story.casefold():
            raise ValueError("Gold answer not supported by the story")
        results.append(DirectExample(
            f"{split}:{i:05d}", skill, prompt, answer))
    return results


def run(*, folder: Path, out: Path, seed: int = 707,
        stages: tuple[tuple[int, int, float, float], ...] = (
            (160, 250, 0.05, 2e-4),
            (800, 650, 0.15, 1e-4),
            (3200, 1600, 0.25, 6e-5),
        ), challenge_count: int = 200) -> dict:
    if (not 1 <= len(stages) <= 5 or
        any(n % 10 or n < 40 or n > 8000 or steps < 1 or steps > 2500 or
            not 0 <= weight < 1 or not 0 < lr < 0.01
            for n, steps, weight, lr in stages) or
        sum(s[1] for s in stages) > 5000):
        raise ValueError("Bounded curriculum invalid")
    if challenge_count % 10 or not 50 <= challenge_count <= 500:
        raise ValueError("Challenge size must be divisible by ten")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    ckpt, manifest, codec = verify_source(folder)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        base = ByteTransformer(ModelConfig(**ckpt["configuration"]))
        base.load_state_dict(ckpt["model"])
        gold, null = copy.deepcopy(base), copy.deepcopy(base)
    source_ids = torch.tensor(
        codec.encode((folder / "train.txt").read_text()).ids, dtype=torch.long)
    validation_ids = torch.tensor(
        codec.encode((folder / "valid.txt").read_text()).ids, dtype=torch.long)
    challenge = make_examples("challenge", challenge_count, seed + 1)
    progress_checks = make_examples("challenge", 50, seed + 77)
    train_full = make_examples("train", stages[-1][0], seed)
    if ({row.prompt for row in train_full} & {row.prompt for row in challenge}
        or {row.prompt for row in progress_checks} & {row.prompt for row in challenge}
        or {row.prompt for row in train_full} & {row.prompt for row in progress_checks}):
        raise ValueError("Training/progress/final challenge question overlap")
    if set(TRAIN_PEOPLE) & set(CHALLENGE_PEOPLE) or (
        set(TRAIN_OBJECTS) & set(CHALLENGE_OBJECTS)):
        raise ValueError("Challenge person/object vocabulary overlaps training")
    def score(model, rows):
        return evaluate_direct(model, codec, rows)
    frozen = {
        "challenge": score(base, challenge),
        "language_loss": round(heldout(base, validation_ids,
                                       windows=64, seq_len=128), 6),
    }
    history = []
    start = time.monotonic()
    for ordinal, (n, steps, text_weight, lr) in enumerate(stages, 1):
        rows = train_full[:n]  # prevents generating differently as stages expand
        shuffled = shuffle_answers_generic(rows, seed + ordinal * 119)
        encoded_gold = encode(rows, codec, base.cfg.max_seq_len)
        encoded_null = encode(shuffled, codec, base.cfg.max_seq_len)
        buckets = {skill: [i for i, row in enumerate(rows) if row.skill == skill]
                   for skill in SKILL_NAMES}
        if any(not bucket for bucket in buckets.values()):
            raise ValueError("Missing supervised skill")
        rng = random.Random(seed + ordinal * 107)
        # Four question examples per batch, sampled across all five skills.
        schedule = [
            [rng.choice(buckets[skill])
             for skill in rng.sample(SKILL_NAMES, 4)]
            for _ in range(steps)
        ]
        def adapt(model, examples, arm):
            opt = torch.optim.AdamW(model.parameters(), lr=lr)
            language_rng = torch.Generator().manual_seed(seed + ordinal * 883)
            model.train()
            last = None
            for step, indices in enumerate(schedule, 1):
                quiz_loss = answer_loss(model, [examples[i] for i in indices])
                if text_weight:
                    x, y = make_batch(source_ids, 2, 64, language_rng)
                    _, lm_loss = model(x, y)
                    loss = ((1 - text_weight) * quiz_loss +
                            text_weight * lm_loss)
                else:
                    loss = quiz_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                last = round(float(quiz_loss.detach()), 6)
            return last
        gold_loss = adapt(gold, encoded_gold, "gold")
        null_loss = adapt(null, encoded_null, "shuffled")
        # A separate progress-check set is not part of the final exam.
        # Final challenge is first evaluated AFTER all training has completed.
        preview = progress_checks
        record = {
            "stage": ordinal, "train_examples": n, "steps_per_arm": steps,
            "learning_rate": lr, "language_rehearsal_weight": text_weight,
            "gold_answer_loss": gold_loss, "shuffled_answer_loss": null_loss,
            "gold_train_preview": score(gold, rows[:min(100,len(rows))]),
            "gold_challenge_preview": score(gold, preview),
            "shuffled_challenge_preview": score(null, preview),
            "gold_language_loss": round(
                heldout(gold, validation_ids, windows=64, seq_len=128), 6),
        }
        history.append(record)
        print(json.dumps({"stage": ordinal, "train": n, "gold_train_accuracy":
                          record["gold_train_preview"]["accuracy"],
                          "challenge_preview_accuracy":
                          record["gold_challenge_preview"]["accuracy"],
                          "text_loss": record["gold_language_loss"]}), flush=True)
    final_gold = score(gold, challenge)
    final_null = score(null, challenge)
    report = {
        "schema": 1, "status": "experimental_not_promoted",
        "source_training_run": 41572257363, "seed": seed,
        "source_checkpoint_sha256": hashlib.sha256(
            (folder / "checkpoint.pt").read_bytes()).hexdigest(),
        "source_manifest_sha256": manifest["train_sha256"],
        "source_parameters": base.num_parameters,
        "same_pretrained_from_scratch_initial_weights": True,
        "same_question_batches_and_public_text_rehearsal": True,
        "train_examples_at_last_stage": stages[-1][0],
        "challenge_examples": len(challenge),
        "progress_check_examples_excluded_from_final": len(progress_checks),
        "train_person_names": list(TRAIN_PEOPLE),
        "challenge_person_names": list(CHALLENGE_PEOPLE),
        "train_objects": list(TRAIN_OBJECTS),
        "challenge_objects": list(CHALLENGE_OBJECTS),
        "challenge_templates_unseen": True,
        "training_gold_sha256": hashlib.sha256(
            "\n".join(r.prompt + " " + r.answer for r in train_full).encode()
        ).hexdigest(),
        "challenge_gold_sha256": hashlib.sha256(
            "\n".join(r.prompt + " " + r.answer for r in challenge).encode()
        ).hexdigest(),
        "frozen": frozen, "stages": history,
        "final_gold": final_gold, "final_shuffled": final_null,
        "verified_language_loss": round(
            heldout(gold, validation_ids, windows=64, seq_len=128), 6),
        "shuffled_language_loss": round(
            heldout(null, validation_ids, windows=64, seq_len=128), 6),
        "story_samples": generate_samples(gold, codec, seed,
                                          new_tokens=64),
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "limitations": [
            "Original synthetic fact generator; not independent natural annotations",
            "Training and challenge use disjoint names/objects/templates, shared skill vocabulary",
            "Separate 50-case progress set is excluded from the 200-case final challenge",
            "Exact greedy answers may penalize correct paraphrases",
            "One random seed and one dataset; no claim of cognitive maturity",
            "No model promotion without replicated and externally validated gains",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "compositional-results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for name, model in (("verified", gold), ("shuffled", null)):
        torch.save({
            "schema": 4, "experimental_only": True, "arm": name,
            "method": "compositional_question_answer",
            "source_checkpoint_sha256": report["source_checkpoint_sha256"],
            "configuration": asdict(model.cfg),
            "model": model.state_dict(),
        }, out / f"{name}-research.pt")
    print(json.dumps({
        "frozen": frozen["challenge"]["correct"],
        "gold": final_gold["correct"], "shuffled": final_null["correct"],
        "total": len(challenge),
        "by_skill_gold": final_gold["by_skill"],
        "story_loss_frozen": frozen["language_loss"],
        "story_loss_gold": report["verified_language_loss"],
        "time_seconds": report["elapsed_seconds"],
    }, indent=2), flush=True)
    return report


def shuffle_answers_generic(rows: list[DirectExample], seed: int) -> list[DirectExample]:
    """Shuffle within each of the five skills, preserving answer frequencies."""
    rng = random.Random(seed)
    answer_pools = {
        skill: [r.answer for r in rows if r.skill == skill]
        for skill in SKILL_NAMES
    }
    for skill, values in answer_pools.items():
        prior = values[:]
        for _ in range(20):
            rng.shuffle(values)
            if values != prior:
                break
        if values == prior:
            raise ValueError("Cannot shuffle the control labels for " + skill)
    at = {skill: 0 for skill in SKILL_NAMES}
    control = []
    for row in rows:
        n = at[row.skill]
        at[row.skill] += 1
        control.append(DirectExample(row.id, row.skill, row.prompt,
                                     answer_pools[row.skill][n]))
    return control


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("artifacts/compositional"))
    p.add_argument("--seed", type=int, default=707)
    a = p.parse_args()
    run(folder=a.checkpoint_dir, out=a.out, seed=a.seed)


if __name__ == "__main__":
    main()
