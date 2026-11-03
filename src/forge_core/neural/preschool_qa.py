"""Forge Seed v0.7 supervised question-answer curriculum: research only.

Two branches inherit the same randomly initialized-and-TinyStories-trained
Forge Seed. Correct-answer teaching is compared with balanced shuffled answers,
using identical public-language rehearsal and optimization opportunities.
No personal information, pretrained teacher weights, or unverified "why" labels.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.preschool_eval import build_prompts, forced_choice
from forge_core.neural.seed_v07 import heldout, generate_samples
from forge_core.neural.train import make_batch

TRAIN_NAMES = ("Nora", "Owen", "Ella", "Finn", "Sara", "June",
               "Liam", "Arlo", "Zoe", "Hugo", "Ruby", "Will")
TRANSFER_NAMES = (("Uma", "Pia"), ("Quinn", "Rex"), ("Vera", "Otis"),
                  ("Nico", "Bea"), ("Cleo", "Eli"))
COLORS = ("red", "blue", "green", "yellow", "pink", "white",
          "black", "purple", "orange", "brown")
PLACES = ("under", "beside", "inside", "outside", "above",
          "below", "behind", "near", "between", "around")
COUNTS = ("one", "two", "three", "four", "five", "six",
          "seven", "eight", "nine", "ten")
SKILLS = ("color", "recipient", "location", "count")
TRAIN_OBJECTS = ("scarf", "hat", "kite", "book", "toy", "cup", "ball", "coat")


@dataclass(frozen=True)
class Task:
    id: str
    skill: str
    prompt: str
    answer: str


def question_prompt(story: str, question: str, correct: str,
                    distractor: str, flip: bool) -> tuple[str, str]:
    if correct == distractor:
        raise ValueError("Answers cannot be identical")
    first, second = (distractor, correct) if flip else (correct, distractor)
    return (f"Story: {story}\nQuestion: {question}\n"
            f"A) {first}\nB) {second}\nAnswer:",
            "B" if flip else "A")


def training_tasks(count: int = 1600, seed: int = 2718) -> list[Task]:
    if not 16 <= count <= 8000 or count % 8:
        raise ValueError("Count must be 16..8000 and divisible by 8")
    rng = random.Random(seed)
    result: list[Task] = []
    for index in range(count):
        skill = SKILLS[index % 4]
        a, b = rng.sample(TRAIN_NAMES, 2)
        flip = (index // 4) % 2 == 1
        if skill == "color":
            c1, c2 = rng.sample(COLORS, 2)
            thing1, thing2 = rng.sample(TRAIN_OBJECTS, 2)
            if index % 3 == 0:
                story = f"{a} put on a {c1} {thing1}. {b} put on a {c2} {thing2}."
                question = f"What was the color of {a}'s {thing1}?"
            else:
                story = f"{a} carried a {c1} {thing1}, while {b} carried a {c2} {thing2}."
                question = f"Which color belonged to {a}'s {thing1}?"
            good, bad = c1, c2
        elif skill == "recipient":
            thing = rng.choice(TRAIN_OBJECTS)
            if index % 3 == 1:
                story = f"{a} gave the {thing} to {b}."
                question = f"Who got the {thing}?"
            else:
                story = f"{a} passed the {thing} over to {b}."
                question = f"Who was given the {thing}?"
            good, bad = b, a
        elif skill == "location":
            where, other = rng.sample(PLACES, 2)
            thing = rng.choice(TRAIN_OBJECTS)
            story = f"{a} put the {thing} {where} the wooden box."
            question = f"Where did {a} put the {thing}?"
            good, bad = where, other
        else:
            n1, n2 = rng.sample(COUNTS, 2)
            story = f"{a} picked up {n1} pebbles. {b} picked up {n2} pebbles."
            question = f"How many pebbles did {a} pick up?"
            good, bad = n1, n2
        prompt, answer = question_prompt(story, question, good, bad, flip)
        result.append(Task(f"train:{index:04d}", skill, prompt, answer))
    return result


def transfer_tasks() -> list[Task]:
    """Forty mirrored transfer questions, with unseen people and phrasings."""
    result = []
    for i, (a, b) in enumerate(TRANSFER_NAMES):
        c1, c2 = COLORS[i], COLORS[i + 5]
        loc1, loc2 = PLACES[i], PLACES[i + 5]
        n1, n2 = COUNTS[i], COUNTS[i + 5]
        records = (
            ("color",
             f"{a} found a {c1} backpack. {b} showed a {c2} basket.",
             f"What color backpack did {a} find?", c1, c2),
            ("recipient",
             f"{b} received a flower from {a}.",
             "Who was handed the flower?", b, a),
            ("location",
             f"The kitten rested {loc1} the fence when {a} arrived.",
             "Where was the kitten resting?", loc1, loc2),
            ("count",
             f"{a} collected {n1} buttons but {b} collected {n2}.",
             f"How many buttons did {a} collect?", n1, n2),
        )
        for j, (skill, story, question, good, bad) in enumerate(records):
            for mirror in (0, 1):
                prompt, answer = question_prompt(story, question, good, bad,
                                                bool(mirror))
                result.append(Task(f"transfer:{i}:{j}:{mirror}", skill,
                                   prompt, answer))
    assert len(result) == 40
    return result


def shuffled_labels(tasks: list[Task], seed: int) -> list[Task]:
    """Keep exact A/B counts within each skill but destroy the answer mapping."""
    rng = random.Random(seed)
    labels: dict[str, list[str]] = {}
    for item in tasks:
        labels.setdefault(item.skill, []).append(item.answer)
    for key, values in labels.items():
        initial = values[:]
        for _ in range(40):
            rng.shuffle(values)
            if values != initial:
                break
        if values == initial:
            raise ValueError(f"Cannot shuffle {key} answers")
    cursor = {key: 0 for key in labels}
    output = []
    for item in tasks:
        k = cursor[item.skill]
        cursor[item.skill] += 1
        output.append(Task(item.id, item.skill, item.prompt, labels[item.skill][k]))
    return output


def encode_tasks(tasks: list[Task], codec: Tokenizer,
                 context: int) -> list[tuple[list[int], int]]:
    encoded = []
    for item in tasks:
        ids = codec.encode(item.prompt).ids
        if not 1 <= len(ids) <= context:
            raise ValueError(f"Task {item.id} exceeds model context")
        encoded.append((ids, 0 if item.answer == "A" else 1))
    return encoded


def classify_batch(model: ByteTransformer,
                   batch: list[tuple[list[int], int]],
                   choice_ids: tuple[int, int]) -> torch.Tensor:
    maxlen = max(len(ids) for ids, _ in batch)
    x = torch.zeros((len(batch), maxlen), dtype=torch.long)
    positions = []
    targets = []
    for i, (ids, answer) in enumerate(batch):
        x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        positions.append(len(ids) - 1)
        targets.append(answer)
    # Causal attention guarantees right-padding cannot affect the logits
    # from each real final prompt token. No special pad embedding needed.
    logits, _ = model(x)
    last = logits[torch.arange(len(batch)), torch.tensor(positions)]
    return F.cross_entropy(last[:, list(choice_ids)],
                           torch.tensor(targets, dtype=torch.long))


def _one_arm(model: ByteTransformer, samples: list[tuple[list[int], int]],
             original_inputs: list[tuple[list[int], int]],
             schedule: list[list[int]], language: torch.Tensor,
             choice_ids: tuple[int, int], seed: int, lr: float,
             text_weight: float, eval_every: int, label: str) -> list[dict]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed + 811)
    model.train()
    curve = []
    for step, indexes in enumerate(schedule, 1):
        batch = [samples[j] for j in indexes]
        if any(samples[j][0] != original_inputs[j][0] for j in indexes):
            raise AssertionError("Control must use exact same question inputs")
        quiz_loss = classify_batch(model, batch, choice_ids)
        if text_weight:
            x, y = make_batch(language, 2, 64, g)
            _, lm_loss = model(x, y)
            loss = (1 - text_weight) * quiz_loss + text_weight * lm_loss
        else:
            loss = quiz_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % eval_every == 0 or step == len(schedule):
            point = {"step": step,
                     "combined_train_loss": round(float(loss.detach()), 5),
                     "quiz_train_loss": round(float(quiz_loss.detach()), 5)}
            curve.append(point)
            print(json.dumps({"arm": label, "progress": point}), flush=True)
    return curve


def _score(model: ByteTransformer, codec: Tokenizer, data: list[Task]) -> dict:
    return forced_choice(model, codec, [
        {"id": row.id, "skill": row.skill, "prompt": row.prompt, "gold": row.answer}
        for row in data
    ])


def verify_source(folder: Path) -> tuple[dict, dict, Tokenizer]:
    manifest_path = folder / "source-manifest.json"
    tokenizer_path = folder / "tokenizer.json"
    checkpoint_path = folder / "checkpoint.pt"
    if not all(p.is_file() for p in
               (manifest_path, tokenizer_path, checkpoint_path,
                folder / "train.txt", folder / "valid.txt")):
        raise ValueError("Missing approved source checkpoint artifact files")
    manifest = json.loads(manifest_path.read_text())
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if ckpt.get("schema") != 3:
        raise ValueError("Not a Forge v0.7 original-model checkpoint")
    if ckpt.get("revision") != manifest.get("revision") or (
        manifest.get("revision") !=
        "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
    ):
        raise ValueError("Source revision mismatch")
    if manifest.get("license") != "cdla-sharing-1.0":
        raise ValueError("Unreviewed source license")
    for part in ("train", "valid"):
        digest = hashlib.sha256((folder / f"{part}.txt").read_bytes()).hexdigest()
        if manifest.get(f"{part}_sha256") != digest or ckpt.get(f"{part}_sha256") != digest:
            raise ValueError(f"Modified {part} source corpus")
    if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != ckpt.get("tokenizer_sha256"):
        raise ValueError("Tokenizer differs from pretrained-from-scratch model")
    return ckpt, manifest, Tokenizer.from_file(str(tokenizer_path))


def research(*, folder: Path, out: Path, steps: int = 1400, seed: int = 913,
             count: int = 1600, batch_size: int = 4, lr: float = 0.00008,
             text_weight: float = 0.25, eval_every: int = 350) -> dict:
    if not 1 <= steps <= 4000 or not 1 <= batch_size <= 8:
        raise ValueError("Experiment exceeds bounded CPU training limits")
    if not 16 <= count <= 8000 or count % 8:
        raise ValueError("Curriculum count must be multiple of eight")
    if not 0 <= text_weight < 1 or not 0 < lr < 0.01 or eval_every < 1:
        raise ValueError("Invalid loss weight or learning rate")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    ckpt, manifest, codec = verify_source(folder)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        frozen = ByteTransformer(ModelConfig(**ckpt["configuration"]))
        frozen.load_state_dict(ckpt["model"])
        gold = copy.deepcopy(frozen)
        null = copy.deepcopy(frozen)
    a_ids, b_ids = codec.encode("A").ids, codec.encode("B").ids
    if len(a_ids) != 1 or len(b_ids) != 1 or a_ids == b_ids:
        raise ValueError("Tokenizer does not provide distinct single-token choice labels")
    choices = (a_ids[0], b_ids[0])
    tasks = training_tasks(count, seed)
    control = shuffled_labels(tasks, seed + 100)
    if [x.prompt for x in tasks] != [x.prompt for x in control]:
        raise AssertionError("Control questions differ from supervised questions")
    for skill in SKILLS:
        if sorted(x.answer for x in tasks if x.skill == skill) != sorted(
                x.answer for x in control if x.skill == skill):
            raise AssertionError(f"Control labels changed class balance for {skill}")
    rng = random.Random(seed + 311)
    # Sample exactly one example from each skill in every group of four.
    bins = {skill: [i for i, x in enumerate(tasks) if x.skill == skill]
            for skill in SKILLS}
    order = [rng.sample(range(4), 4) for _ in range(steps)]
    schedule = [[rng.choice(bins[SKILLS[j]]) for j in group[:batch_size]]
                for group in order] if batch_size <= 4 else [
        [rng.choice(bins[SKILLS[j % 4]]) for j in rng.sample(range(8), batch_size)]
        for _ in range(steps)]
    original_inputs = encode_tasks(tasks, codec, frozen.cfg.max_seq_len)
    shuffled_inputs = encode_tasks(control, codec, frozen.cfg.max_seq_len)
    text_tokens = torch.tensor(codec.encode((folder / "train.txt").read_text()).ids,
                               dtype=torch.long)
    heldout_tokens = torch.tensor(codec.encode((folder / "valid.txt").read_text()).ids,
                                  dtype=torch.long)
    heldout_eval = transfer_tasks()
    benchmark = build_prompts()
    initial = {
        "legacy_40": _score(frozen, codec, [
            Task(x["id"], x["skill"], x["prompt"], x["gold"])
            for x in benchmark]),
        "transfer_40": _score(frozen, codec, heldout_eval),
        "language_loss": round(heldout(frozen, heldout_tokens, windows=64,
                                       seq_len=128), 6),
    }
    start = time.monotonic()
    gold_curve = _one_arm(gold, original_inputs, original_inputs, schedule,
                          text_tokens, choices, seed, lr, text_weight,
                          eval_every, "verified")
    null_curve = _one_arm(null, shuffled_inputs, original_inputs, schedule,
                          text_tokens, choices, seed, lr, text_weight,
                          eval_every, "shuffled")
    def evaluate_arm(model):
        return {
            "legacy_40": _score(model, codec, [
                Task(x["id"], x["skill"], x["prompt"], x["gold"])
                for x in benchmark]),
            "transfer_40": _score(model, codec, heldout_eval),
            "language_loss": round(heldout(model, heldout_tokens, windows=64,
                                           seq_len=128), 6),
        }
    report = {
        "schema": 1, "status": "experimental_not_promoted",
        "source_run_id": 41572257363, "source_step": ckpt["step"],
        "source_checkpoint_sha256": hashlib.sha256(
            (folder / "checkpoint.pt").read_bytes()).hexdigest(),
        "source_train_sha256": manifest["train_sha256"],
        "source_valid_sha256": manifest["valid_sha256"],
        "source_tokenizer_sha256": ckpt["tokenizer_sha256"],
        "model_parameters": frozen.num_parameters,
        "seed": seed, "training_examples": len(tasks),
        "adaptation_steps_per_arm": steps, "batch_size": batch_size,
        "learning_rate": lr, "shared_language_loss_weight": text_weight,
        "same_initial_weights": True, "same_qa_minibatches": True,
        "same_public_text_minibatches": True,
        "baseline": initial,
        "verified_teaching": evaluate_arm(gold),
        "shuffled_teaching": evaluate_arm(null),
        "gold_curve": gold_curve, "shuffled_curve": null_curve,
        "fixed_seed_story_samples": generate_samples(gold, codec, seed,
                                                     new_tokens=64),
        "training_seconds": round(time.monotonic() - start, 2),
        "limitations": [
            "Original synthetically generated QA, not independently annotated natural language",
            "Names differ between training and both tests, but tasks share underlying relations",
            "Forty mirrored choices are not forty independent story questions",
            "Shuffled-label null is not an equal-compute text-only continuation baseline",
            "Instruction following in free-form chat is NOT measured by A/B forced choices",
            "No skill or checkpoint promotion based on one seed or one test set",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(report, indent=2,
                                                 ensure_ascii=False) + "\n")
    (out / "curriculum-manifest.json").write_text(json.dumps({
        "schema": 1, "origin": "Forge Core original deterministic synthetic QA",
        "seed": seed, "count": count,
        "train_ids_sha256": hashlib.sha256(
            "\n".join(x.id + " " + x.prompt + " " + x.answer
                      for x in tasks).encode()).hexdigest(),
        "train_names": list(TRAIN_NAMES),
        "transfer_eval_names": [x for pair in TRANSFER_NAMES for x in pair],
        "public_checkpoint": report["source_checkpoint_sha256"],
        "user_data_included": False,
    }, indent=2) + "\n")
    for name, model in (("verified", gold), ("shuffled", null)):
        torch.save({
            "schema": 4, "experimental_only": True,
            "method": name,
            "base_checkpoint_sha256": report["source_checkpoint_sha256"],
            "configuration": asdict(model.cfg),
            "model": model.state_dict(), "adaptation_steps": steps,
        }, out / f"{name}-research.pt")
    print(json.dumps({
        "baseline": initial, "verified": report["verified_teaching"],
        "shuffled": report["shuffled_teaching"],
        "wall_seconds": report["training_seconds"],
    }, indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("artifacts/qa-research"))
    p.add_argument("--steps", type=int, default=1400)
    p.add_argument("--examples", type=int, default=1600)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=913)
    p.add_argument("--text-weight", type=float, default=0.25)
    a = p.parse_args()
    research(folder=a.checkpoint_dir, out=a.out, steps=a.steps,
             seed=a.seed, count=a.examples, batch_size=a.batch,
             text_weight=a.text_weight,
             eval_every=max(1, a.steps // 4))


if __name__ == "__main__":
    main()
