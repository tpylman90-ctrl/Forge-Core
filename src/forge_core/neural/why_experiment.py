"""Controlled A/B test of Forge Seed's first linguistic 'why' teaching.

Two independently adapted models start from the SAME scratch-trained Forge
checkpoint, receive the SAME prompts, gradient updates, schedule and optimizer.
The only intervention is correct verified labels versus class-balanced shuffled
labels. A frozen base model is scored separately. This is a tiny proof of
experimental plumbing, not evidence of broad syntax or reasoning competence.

NO personal data. NO pretrained model. NO self-labeled training.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from forge_core.learning.linguistic_why import build_probes, validate_splits
from forge_core.neural.archive import verify_archive
from forge_core.neural.experiment import evaluate
from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.train import make_batch

CHOICES = (ord("A"), ord("B"))


@dataclass(frozen=True)
class ChoiceExample:
    id: str
    skill: str
    prompt: str
    answer: str


def build_examples(split: str) -> list[ChoiceExample]:
    """Prompt text never contains gold labels or rule explanations."""
    probes = build_probes(split)
    examples: list[ChoiceExample] = []
    for p in probes:
        if p.skill == "semantic_roles":
            for side, sentence in (("a", p.sentence_a), ("b", p.sentence_b)):
                for who in ("agent", "patient"):
                    agent, patient = p.gold[side]["agent"], p.gold[side]["patient"]
                    # Mix where the agent/patient appears so options do not leak
                    # a stable answer position. Answers come only from gold.
                    variant = int(p.id.split(":")[-1]) + (side == "b") + (who == "patient")
                    options = (agent, patient) if variant % 2 == 0 else (patient, agent)
                    correct = p.gold[side][who]
                    answer = "A" if options[0] == correct else "B"
                    question = "Who performs the action?" if who == "agent" else "Who receives the action?"
                    prompt = (
                        f"Sentence: {sentence}\nQuestion: {question}\n"
                        f"A) {options[0]}\nB) {options[1]}\nAnswer: "
                    )
                    examples.append(ChoiceExample(
                        f"{p.id}:{side}:{who}", p.skill, prompt, answer))
        elif p.skill == "subject_verb_agreement":
            prompt = (
                "Question: Which sentence follows standard written English subject-verb agreement?\n"
                f"A) {p.sentence_a}\nB) {p.sentence_b}\nAnswer: "
            )
            answer = p.gold["grammatical"].upper()
            examples.append(ChoiceExample(p.id, p.skill, prompt, answer))
        elif p.skill == "negation":
            prompt = (
                "Question: Which sentence says the event did not happen?\n"
                f"A) {p.sentence_a}\nB) {p.sentence_b}\nAnswer: "
            )
            side = next(side for side in ("a", "b")
                        if p.gold[side]["event_polarity"] == "negated")
            examples.append(ChoiceExample(p.id, p.skill, prompt, side.upper()))
        else:
            raise ValueError(f"Unsupported gold skill {p.skill}")
    if any(x.answer not in ("A", "B") for x in examples):
        raise ValueError("Bad gold label")
    if len({x.id for x in examples}) != len(examples):
        raise ValueError("Duplicate examples")
    return examples


def shuffled_control(examples: list[ChoiceExample], seed: int) -> list[ChoiceExample]:
    """Preserve per-skill label frequencies while breaking prompt/label links."""
    rng = random.Random(seed)
    labels_by_skill: dict[str, list[str]] = {}
    for ex in examples:
        labels_by_skill.setdefault(ex.skill, []).append(ex.answer)
    for skill, labels in labels_by_skill.items():
        original = labels[:]
        for _ in range(30):
            rng.shuffle(labels)
            if labels != original:
                break
        if labels == original:
            raise ValueError(f"Cannot create label control for {skill}")
    positions = {skill: 0 for skill in labels_by_skill}
    result = []
    for ex in examples:
        index = positions[ex.skill]
        positions[ex.skill] += 1
        result.append(ChoiceExample(ex.id, ex.skill, ex.prompt,
                                    labels_by_skill[ex.skill][index]))
    return result


@torch.no_grad()
def score_choices(model: ByteTransformer, examples: list[ChoiceExample]) -> dict:
    model.eval()
    aggregate: dict[str, dict] = {}
    total_nll = 0.0
    correct_total = 0
    for ex in examples:
        raw = ex.prompt.encode("utf-8")
        if len(raw) > model.cfg.max_seq_len:
            raise ValueError("Probe exceeds model context length")
        logits, _ = model(torch.tensor([list(raw)], dtype=torch.long))
        # Restrict normalization to the two permitted answers.
        pair = logits[0, -1, [CHOICES[0], CHOICES[1]]]
        logp = F.log_softmax(pair, dim=0)
        expected = 0 if ex.answer == "A" else 1
        correct = int(int(pair.argmax()) == expected)
        nll = float(-logp[expected])
        category = aggregate.setdefault(ex.skill, {"correct": 0, "total": 0, "nll": 0.0})
        category["correct"] += correct
        category["total"] += 1
        category["nll"] += nll
        correct_total += correct
        total_nll += nll
    by_skill = {skill: {"accuracy": round(rec["correct"] / rec["total"], 4),
                        "correct": rec["correct"], "total": rec["total"],
                        "mean_label_nll": round(rec["nll"] / rec["total"], 5)}
                for skill, rec in sorted(aggregate.items())}
    return {"accuracy": round(correct_total / len(examples), 4),
            "correct": correct_total, "total": len(examples),
            "mean_label_nll": round(total_nll / len(examples), 5),
            "by_skill": by_skill}


def _adapt(model: ByteTransformer, examples: list[ChoiceExample],
           labels: list[ChoiceExample], schedule: list[list[int]],
           lr: float, language_batches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
           language_weight: float = 0.0) -> float:
    """Same prompts, same public-language examples, same batch count in both arms.

    Only the linguistic gold labels differ. The optional public-language
    auxiliary objective can protect general next-byte prediction, but cannot
    by itself prove semantic understanding.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    last = float("nan")
    for step_index, sample_indices in enumerate(schedule):
        opt.zero_grad(set_to_none=True)
        for i in sample_indices:
            ex, label = examples[i], labels[i]
            assert ex.id == label.id and ex.prompt == label.prompt
            prefix = torch.tensor([list(ex.prompt.encode("utf-8"))], dtype=torch.long)
            logits, _ = model(prefix)
            pair = logits[0, -1, [CHOICES[0], CHOICES[1]]].unsqueeze(0)
            y = torch.tensor([0 if label.answer == "A" else 1], dtype=torch.long)
            task_loss = F.cross_entropy(pair, y)
            ((1.0 - language_weight) * task_loss / len(sample_indices)).backward()
            last = float(task_loss.detach())
        if language_batches is not None:
            x, y = language_batches[step_index]
            _, language_loss = model(x, y)
            (language_weight * language_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return last


def run_experiment(*, checkpoint_path: Path, validation_path: Path,
                   steps: int = 100, batch: int = 2, seed: int = 1234,
                   lr: float = 5e-5, validation_windows: int = 16,
                   output_dir: Path | None = None,
                   language_train_path: Path | None = None,
                   language_weight: float = 0.0) -> dict:
    if not 1 <= steps <= 1000 or not 1 <= batch <= 8 or not 0 < lr <= 0.01:
        raise ValueError("Invalid bounded adaptation settings")
    if not 2 <= validation_windows <= 64:
        raise ValueError("Invalid validation window count")
    if not 0 <= language_weight < 1:
        raise ValueError("Language loss weight must be in [0, 1)")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if ckpt.get("schema") != 2 or "model" not in ckpt or "valid_sha256" not in ckpt:
        raise ValueError("Checkpoint lacks Forge Core provenance metadata")
    raw = validation_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ckpt["valid_sha256"]:
        raise ValueError("Validation text differs from source checkpoint")
    train_probes, eval_probes = build_probes("train"), build_probes("eval")
    validate_splits(train_probes, eval_probes)
    train = build_examples("train")
    evaluation = build_examples("eval")
    if {x.id for x in train} & {x.id for x in evaluation}:
        raise ValueError("Probe leakage across splits")
    control = shuffled_control(train, seed + 117)
    if sorted(x.answer for x in train) != sorted(x.answer for x in control):
        raise ValueError("Control must preserve overall label frequencies")
    cfg = ModelConfig(**ckpt["config"])
    if max(len(x.prompt.encode("utf-8")) for x in train + evaluation) > cfg.max_seq_len:
        raise ValueError("Model too small for these language probes")
    rng = random.Random(seed)
    schedule = [[rng.randrange(len(train)) for _ in range(batch)] for _ in range(steps)]
    language_batches = None
    if language_weight:
        if language_train_path is None:
            raise ValueError("Public source training text is required for language mixing")
        train_bytes = language_train_path.read_bytes()
        if hashlib.sha256(train_bytes).hexdigest() != ckpt.get("train_sha256"):
            raise ValueError("Public training text differs from source checkpoint")
        seq = min(64, cfg.max_seq_len)
        if len(train_bytes) <= seq:
            raise ValueError("Training text too small for language mixing")
        generator = torch.Generator().manual_seed(seed + 909)
        tensor_data = torch.tensor(list(train_bytes), dtype=torch.long)
        language_batches = [make_batch(tensor_data, batch, seq, generator)
                            for _ in range(steps)]
    start = time.monotonic()
    # Both arms clone the SAME pretrained-from-scratch Forge checkpoint
    # (not a third-party teacher). Optimizers are freshly initialized.
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        frozen = ByteTransformer(cfg)
        frozen.load_state_dict(ckpt["model"])
        gold_model = copy.deepcopy(frozen)
        null_model = copy.deepcopy(frozen)
    base_eval = score_choices(frozen, evaluation)
    base_train = score_choices(frozen, train)
    last_gold = _adapt(gold_model, train, train, schedule, lr,
                       language_batches, language_weight)
    last_null = _adapt(null_model, train, control, schedule, lr,
                       language_batches, language_weight)
    score_gold = score_choices(gold_model, evaluation)
    score_null = score_choices(null_model, evaluation)
    train_gold = score_choices(gold_model, train)
    train_null = score_choices(null_model, train)
    valid = torch.tensor(list(raw), dtype=torch.long)
    seq_len = min(128, cfg.max_seq_len)
    if valid.numel() <= seq_len:
        raise ValueError("Validation file too short")
    # Independently measure distributional-language regression (not just A/B)
    window_batches = validation_windows // 2
    stats = {
        "frozen": evaluate(frozen, valid, 2, seq_len, window_batches, seed, torch.device("cpu")),
        "verified_labels": evaluate(gold_model, valid, 2, seq_len, window_batches, seed, torch.device("cpu")),
        "shuffled_labels": evaluate(null_model, valid, 2, seq_len, window_batches, seed, torch.device("cpu")),
    }
    summary = {
        "schema": 1, "status": "experimental_not_promoted",
        "source_checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "source_step": ckpt["step"], "validation_sha256": ckpt["valid_sha256"],
        "model_parameters": frozen.num_parameters,
        "seed": seed, "steps_per_arm": steps, "batch_size": batch, "learning_rate": lr,
        "public_language_loss_weight": language_weight,
        "identical_language_minibatches": bool(language_batches),
        "prompt_examples_train": len(train), "prompt_examples_eval": len(evaluation),
        "identical_schedule": True, "identical_initial_weights": True,
        "same_number_of_prompts_and_optimizer_updates": True,
        "shuffled_control_preserves_task_label_counts": True,
        "train_gold": train_gold, "train_shuffled": train_null,
        "eval_frozen": base_eval, "eval_gold": score_gold, "eval_shuffled": score_null,
        "train_frozen": base_train,
        "heldout_next_byte_loss": {k: round(v, 6) for k, v in stats.items()},
        "last_gold_batch_loss": round(last_gold, 6),
        "last_shuffled_batch_loss": round(last_null, 6),
        "wall_seconds": round(time.monotonic() - start, 2),
        "limitations": [
            "Tiny hand-written templates; eval nouns/verbs unseen but construction templates overlap",
            "Single training seed; no external grammar corpus; not proof of generalized language or causes",
            "Control uses shuffled labels rather than an equal-compute text-only language curriculum",
            "Equal updates/prompt budgets, not verified equal effective gradient information",
            "Neither adapted model is approved for release or Android integration",
        ],
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "why-experiment.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for model_name, model in (("verified-labels", gold_model), ("shuffled-labels", null_model)):
            torch.save({
                "config": asdict(cfg), "source_checkpoint_sha256": summary["source_checkpoint_sha256"],
                "model": model.state_dict(), "experiment_only": True,
                "method": model_name, "steps": steps, "seed": seed,
            }, output_dir / f"{model_name}.pt")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True,
                   help="Trusted, downloaded Forge Core public TinyStories checkpoint artifact folder")
    p.add_argument("--out", type=Path, default=Path("artifacts/linguistic-why"))
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--language-weight", type=float, default=0.0,
                   help="Shared public-language next-byte loss contribution, 0..1")
    args = p.parse_args()
    audit = verify_archive(args.archive)
    result = run_experiment(
        checkpoint_path=args.archive / "checkpoint.pt",
        validation_path=args.archive / "valid.txt",
        steps=args.steps, batch=args.batch, seed=args.seed,
        output_dir=args.out,
        language_train_path=args.archive / "train.txt",
        language_weight=args.language_weight,
    )
    result["archive_audit"] = {"status": audit["archive_status"],
                              "source": audit["source"], "license": audit["license"]}
    (args.out / "why-experiment.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
