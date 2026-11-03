"""Direct-answer (not A/B) apprenticeship for a scratch Forge Seed.

Original verified counterfactual cases become short natural answer targets.
A BPE causal model is taught to complete 'Answer: <value>.' while keeping
the pretrained public TinyStories language loss under observation.
Two adaptations share identical source weights, prompts, steps, sampling
schedule and language rehearsal; only correct vs shuffled answers differ.
NO personal data and NO automatic model promotion.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import re
import time

import torch
from torch.nn import functional as F

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.preschool_eval import build_prompts
from forge_core.neural.preschool_qa import (
    SKILLS, Task, transfer_tasks, verify_source,
)
from forge_core.neural.paired_contrasts import paired_tasks
from forge_core.neural.seed_v07 import heldout, generate_samples
from forge_core.neural.train import make_batch


@dataclass(frozen=True)
class DirectExample:
    id: str
    skill: str
    prompt: str
    answer: str


def direct_examples(choices: list[Task]) -> list[DirectExample]:
    """Convert adjacent A/B mirrors into one verified natural answer.

    The two official generators use different mirror ID conventions.
    Validate content instead of assuming any particular ID spelling.
    A mirrored pair is one independent underlying question.
    """
    if not choices or len(choices) % 2:
        raise ValueError("Expected complete mirrored source pairs")
    results = []
    for index in range(0, len(choices), 2):
        first, second = choices[index:index + 2]
        if first.skill != second.skill or first.answer != "A" or second.answer != "B":
            raise ValueError("Corrupt or out-of-order mirrored answer pairs")

        def parse(task: Task):
            if "\nA) " not in task.prompt or "\nB) " not in task.prompt:
                raise ValueError("Missing options in verified question")
            prefix, remainder = task.prompt.split("\nA) ", 1)
            if "\nAnswer:" not in remainder:
                raise ValueError("Missing answer marker")
            opt_a, remainder = remainder.split("\nB) ", 1)
            opt_b = remainder.rsplit("\nAnswer:", 1)[0]
            if not opt_a.strip() or not opt_b.strip() or opt_a == opt_b:
                raise ValueError("Invalid verified answer options")
            return prefix, opt_a, opt_b

        prefix, a, b = parse(first)
        other_prefix, mirrored_a, mirrored_b = parse(second)
        if prefix != other_prefix or a != mirrored_b or b != mirrored_a:
            raise ValueError("Mirrors do not preserve the same underlying fact")
        # Both mirror labels are checked and derive the identical gold word.
        if (a if first.answer == "A" else b) != (
            mirrored_a if second.answer == "A" else mirrored_b):
            raise ValueError("Contradictory gold source labels")
        results.append(DirectExample(
            f"direct:{index // 2:04d}", first.skill,
            prefix + "\nAnswer:", a))

    if len({x.id for x in results}) != len(results):
        raise ValueError("Duplicate direct teaching IDs")
    return results

def shuffle_answers(rows: list[DirectExample], seed: int) -> list[DirectExample]:
    rng = random.Random(seed)
    groups = {skill: [row.answer for row in rows if row.skill == skill]
              for skill in SKILLS}
    for skill, answers in groups.items():
        original = answers[:]
        for _ in range(20):
            rng.shuffle(answers)
            if answers != original:
                break
        if answers == original:
            raise ValueError(f"Cannot create randomized labels for {skill}")
    cursor = {skill: 0 for skill in SKILLS}
    out = []
    for row in rows:
        index = cursor[row.skill]
        cursor[row.skill] += 1
        out.append(DirectExample(row.id, row.skill, row.prompt,
                                 groups[row.skill][index]))
    return out


def encode(rows: list[DirectExample], tokenizer, max_context: int):
    data = []
    for row in rows:
        prefix = tokenizer.encode(row.prompt).ids
        answer = tokenizer.encode(" " + row.answer + ".").ids
        if not prefix or not answer or len(prefix) + len(answer) - 1 > max_context:
            raise ValueError("Direct answer exceeds model context")
        # Last prefix token predicts first answer token. All other prefix
        # positions are excluded from task loss, but supply its context.
        tokens = prefix + answer[:-1]
        labels = ([-100] * (len(prefix) - 1)) + answer
        assert len(tokens) == len(labels)
        data.append((tokens, labels))
    return data


def answer_loss(model: ByteTransformer, examples) -> torch.Tensor:
    longest = max(len(x) for x, _ in examples)
    X = torch.zeros((len(examples), longest), dtype=torch.long)
    Y = torch.full((len(examples), longest), -100, dtype=torch.long)
    for i, (tokens, labels) in enumerate(examples):
        X[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        Y[i, :len(labels)] = torch.tensor(labels, dtype=torch.long)
    logits, _ = model(X)
    return F.cross_entropy(logits.reshape(-1, model.cfg.vocab_size),
                           Y.reshape(-1), ignore_index=-100)


def normalize_answer(answer: str) -> str:
    return re.sub(r"\s+", " ", re.split(r"[.!?\n]", answer.strip(), maxsplit=1)[0]
                  .strip(" \t\"'")).casefold()


@torch.no_grad()
def evaluate_direct(model: ByteTransformer, tokenizer,
                    rows: list[DirectExample], max_tokens: int = 10) -> dict:
    model.eval()
    # Evaluation must support new audited language skills, e.g. giver;
    # the old four-skill fixed dictionary silently blocked expansions.
    scores: dict[str, dict[str, int]] = {}
    examples = []
    successes = 0
    for row in rows:
        prefix = tokenizer.encode(row.prompt).ids
        ids = prefix[:]
        continuation = []
        for _ in range(max_tokens):
            logits, _ = model(torch.tensor([ids[-model.cfg.max_seq_len:]],
                                           dtype=torch.long))
            next_id = int(logits[0, -1].argmax())
            ids.append(next_id)
            continuation.append(next_id)
            if "." in tokenizer.decode(continuation):
                break
        text = tokenizer.decode(continuation)
        correct = int(normalize_answer(text) == normalize_answer(row.answer))
        bucket = scores.setdefault(row.skill, {"correct": 0, "total": 0})
        bucket["total"] += 1
        bucket["correct"] += correct
        successes += correct
        if len(examples) < 8:
            examples.append({"skill": row.skill, "question": row.prompt,
                             "expected": row.answer, "predicted": text,
                             "correct": bool(correct)})
    return {
        "correct": successes, "total": len(rows),
        "accuracy": round(successes / len(rows), 4),
        "by_skill": {skill: {**v, "accuracy": round(v["correct"] / v["total"], 4)}
                     for skill, v in scores.items() if v["total"]},
        "examples": examples, "scoring": "Exact short answer, greedy decoding",
    }


def run(*, folder: Path, out: Path, seed: int = 710,
        stages: tuple[tuple[int,int,float,float],...] = (
            (32, 250, 0.00, 3e-4),
            (128, 500, 0.10, 1.5e-4),
            (512, 1250, 0.20, 7e-5),
        )) -> dict:
    if not stages or len(stages) > 5 or sum(s[1] for s in stages) > 4000:
        raise ValueError("Invalid bounded direct-answer schedule")
    if any(n < 32 or n % 32 or n > 4096 or steps < 1 or
           not 0 <= language_weight < 1 or not 0 < lr < 0.01
           for n, steps, language_weight, lr in stages):
        raise ValueError("Invalid direct-answer stage")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    checkpoint, manifest, codec = verify_source(folder)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        base = ByteTransformer(ModelConfig(**checkpoint["configuration"]))
        base.load_state_dict(checkpoint["model"])
        gold = copy.deepcopy(base)
        shuffled = copy.deepcopy(base)
    train_token_ids = torch.tensor(
        codec.encode((folder / "train.txt").read_text()).ids, dtype=torch.long)
    valid_token_ids = torch.tensor(
        codec.encode((folder / "valid.txt").read_text()).ids, dtype=torch.long)
    legacy = direct_examples([
        Task(x["id"], x["skill"], x["prompt"], x["gold"])
        for x in build_prompts()
    ])
    transfer = direct_examples(transfer_tasks())
    holdout_prompts = {x.prompt for x in legacy + transfer}
    if {x.prompt for x in direct_examples(paired_tasks(stages[-1][0], seed))} & holdout_prompts:
        raise ValueError("Leaked held-out question into direct training")

    def score(model):
        return {
            "original_20": evaluate_direct(model, codec, legacy),
            "transfer_20": evaluate_direct(model, codec, transfer),
            "heldout_story_token_loss": round(
                heldout(model, valid_token_ids, windows=64, seq_len=128), 6),
        }
    frozen_score = score(base)
    start = time.monotonic()
    phases = []
    for stage_index, (n, steps, text_weight, lr) in enumerate(stages, 1):
        teacher = direct_examples(paired_tasks(n, seed))
        null = shuffle_answers(teacher, seed + stage_index * 239)
        if [x.prompt for x in teacher] != [x.prompt for x in null]:
            raise ValueError("Changed prompts in shuffled control")
        teacher_tokens = encode(teacher, codec, base.cfg.max_seq_len)
        null_tokens = encode(null, codec, base.cfg.max_seq_len)
        bins = {skill: [i for i, row in enumerate(teacher) if row.skill == skill]
                for skill in SKILLS}
        rng = random.Random(seed + stage_index * 333)
        order = [[rng.choice(bins[skill]) for skill in rng.sample(SKILLS, 4)]
                 for _ in range(steps)]

        def adapt(model, encoded, arm):
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            source_rng = torch.Generator().manual_seed(seed + stage_index * 71)
            model.train()
            last = 0.0
            for step, indices in enumerate(order, 1):
                cls_loss = answer_loss(model, [encoded[i] for i in indices])
                if text_weight:
                    x, y = make_batch(train_token_ids, 2, 64, source_rng)
                    _, lm_loss = model(x, y)
                    loss = (1 - text_weight) * cls_loss + text_weight * lm_loss
                else:
                    loss = cls_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                last = round(float(cls_loss.detach()), 5)
                if step == steps:
                    print(json.dumps({"stage": stage_index, "arm": arm,
                                      "final_train_answer_loss": last}), flush=True)
            return last

        gold_train_loss = adapt(gold, teacher_tokens, "verified")
        null_train_loss = adapt(shuffled, null_tokens, "shuffled")
        data = {
            "stage": stage_index, "total_source_choice_questions": n,
            "independent_direct_training_questions": len(teacher),
            "optimizer_updates_per_arm": steps,
            "language_rehearsal_weight": text_weight,
            "gold_final_training_answer_loss": gold_train_loss,
            "shuffled_final_training_answer_loss": null_train_loss,
            "gold_training_exact_match": evaluate_direct(gold, codec, teacher),
            "gold": score(gold),
            "shuffled": score(shuffled),
        }
        phases.append(data)
        print(json.dumps({
            "stage_complete": stage_index,
            "training_direct_correct": data["gold_training_exact_match"]["correct"],
            "legacy_correct": data["gold"]["original_20"]["correct"],
            "transfer_correct": data["gold"]["transfer_20"]["correct"],
            "language_loss": data["gold"]["heldout_story_token_loss"],
        }), flush=True)
    final = score(gold)
    control = score(shuffled)
    no_language_regression = final["heldout_story_token_loss"] <= (
        frozen_score["heldout_story_token_loss"] * 1.05)
    # Preliminary diagnostics only; no automatic promotion.
    report = {
        "schema": 1, "research_only": True, "preschool_certified": False,
        "checkpoint_promoted": False,
        "source_training_run": 41572257363,
        "source_checkpoint_sha256": hashlib.sha256(
            (folder / "checkpoint.pt").read_bytes()).hexdigest(),
        "train_sha256": manifest["train_sha256"], "seed": seed,
        "model_parameters": base.num_parameters,
        "source_pretraining_step": checkpoint["step"],
        "source_model_from_random_weights": True,
        "training_type": "direct short answer masked-token supervised learning",
        "data_scope": "original verified synthetic QA plus public TinyStories only",
        "frozen": frozen_score,
        "stages": phases,
        "final_gold": final, "final_shuffled": control,
        "language_loss_within_five_percent": no_language_regression,
        "human_grading_done": False,
        "fixed_story_samples": generate_samples(gold, codec, seed, new_tokens=64),
        "runtime_seconds": round(time.monotonic() - start, 2),
        "limitations": [
            "Synthetic rule-generated tasks with overlapping relationships",
            "Twenty original plus twenty transfer underlying tasks, one seed",
            "Some training data shares construction families with evaluation",
            "Exact-match short-answer scoring, not nuanced correctness",
            "Not an independently established preschool or instruction ability",
            "Shuffled labels are not a matched all-text training continuation",
            "No model promotion without independent later evidence",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "direct-results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for name, model in (("verified", gold), ("shuffled", shuffled)):
        torch.save({
            "schema": 4, "experimental_only": True, "arm": name,
            "source_checkpoint_sha256": report["source_checkpoint_sha256"],
            "model": model.state_dict(), "configuration": asdict(model.cfg),
            "method": "direct_short_answers", "seed": seed,
        }, out / f"{name}-research.pt")
    print(json.dumps({"frozen": frozen_score, "gold": final,
                      "shuffled": control, "language_retention": no_language_regression,
                      "wall_seconds": report["runtime_seconds"]},
                     indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("artifacts/direct-qa"))
    a = p.parse_args()
    run(folder=a.checkpoint_dir, out=a.out)


if __name__ == "__main__":
    main()
