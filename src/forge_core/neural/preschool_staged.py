"""Staged question-answer curriculum for original Forge Seed v0.7.

Research experiment only: tiny repeated foundations → more varied questions →
transfer tests with names and forms never included in adaptation. Compares
gold-rule answers to class-balanced shuffled labels. Public-language rehearsal
is identical in both arms. No promotion or external action.
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
from forge_core.neural.preschool_eval import build_prompts
from forge_core.neural.preschool_qa import (
    Task, SKILLS, _one_arm, _score, encode_tasks, shuffled_labels,
    training_tasks, transfer_tasks, verify_source,
)
from forge_core.neural.seed_v07 import heldout, generate_samples

# Training count, optimizer updates per independent arm, rehearsal fraction,
# AdamW LR. Small repeated examples first. None are personal data.
DEFAULT_STAGES = (
    (16, 200, 0.00, 3e-4),
    (64, 400, 0.10, 1.5e-4),
    (256, 800, 0.20, 8e-5),
    (512, 1200, 0.25, 6e-5),
)


def curriculum(*, folder: Path, output: Path, seed: int = 99,
               stages: tuple[tuple[int, int, float, float], ...] = DEFAULT_STAGES,
               batch_size: int = 4) -> dict:
    if not 1 <= batch_size <= 4 or not 1 <= len(stages) <= 6:
        raise ValueError("Invalid bounded staged-curriculum parameters")
    if any(not (16 <= n <= 8000 and n % 8 == 0 and 1 <= steps <= 2000
                and 0 <= weight < 1 and 0 < lr < 0.01)
           for n, steps, weight, lr in stages):
        raise ValueError("Invalid stage training limits")
    if sum(x[1] for x in stages) > 5000:
        raise ValueError("Too many total optimizer updates")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    checkpoint, manifest, codec = verify_source(folder)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        frozen = ByteTransformer(ModelConfig(**checkpoint["configuration"]))
        frozen.load_state_dict(checkpoint["model"])
        gold_model = copy.deepcopy(frozen)
        shuffled_model = copy.deepcopy(frozen)
    a, b = codec.encode("A").ids, codec.encode("B").ids
    if len(a) != 1 or len(b) != 1 or a == b:
        raise ValueError("Answer labels are not distinct atomic tokens")
    choices = (a[0], b[0])
    raw_train = codec.encode((folder / "train.txt").read_text()).ids
    raw_valid = codec.encode((folder / "valid.txt").read_text()).ids
    train_data = torch.tensor(raw_train, dtype=torch.long)
    valid_data = torch.tensor(raw_valid, dtype=torch.long)
    transfer = transfer_tasks()
    legacy = [Task(x["id"], x["skill"], x["prompt"], x["gold"])
              for x in build_prompts()]

    def score(model: ByteTransformer) -> dict:
        return {
            "legacy_40": _score(model, codec, legacy),
            "transfer_40": _score(model, codec, transfer),
            "heldout_token_loss": round(
                heldout(model, valid_data, windows=64, seq_len=128), 6),
        }

    frozen_scores = score(frozen)
    history = []
    start = time.monotonic()
    for stage_number, (n, steps, weight, lr) in enumerate(stages, start=1):
        # Every subsequent stage includes the previous stage's examples.
        items = training_tasks(n, seed)
        random_labels = shuffled_labels(items, seed + stage_number * 101)
        gold_encoded = encode_tasks(items, codec, gold_model.cfg.max_seq_len)
        shuffled_encoded = encode_tasks(random_labels, codec,
                                         shuffled_model.cfg.max_seq_len)
        if any(x[0] != y[0] for x,y in zip(gold_encoded, shuffled_encoded)):
            raise ValueError("Experimental controls have different prompts")
        # Category-balanced, identical schedule for both model adaptations.
        buckets = {skill: [i for i, item in enumerate(items) if item.skill == skill]
                   for skill in SKILLS}
        rng = random.Random(seed + 900 + stage_number)
        schedule = [
            [rng.choice(buckets[skill])
             for skill in rng.sample(SKILLS, batch_size)]
            for _ in range(steps)
        ]
        same_seed = seed + 1000 * stage_number
        gold_curve = _one_arm(gold_model, gold_encoded, gold_encoded, schedule,
                              train_data, choices, same_seed, lr, weight,
                              max(1, steps // 2), "gold")
        control_curve = _one_arm(
            shuffled_model, shuffled_encoded, gold_encoded, schedule,
            train_data, choices, same_seed, lr, weight,
            max(1, steps // 2), "shuffled")
        phase = {
            "stage": stage_number,
            "training_example_count": n,
            "optimizer_steps_per_arm": steps,
            "language_rehearsal_weight": weight,
            "lr": lr,
            "gold_training_qa": _score(gold_model, codec, items),
            "shuffled_training_qa": _score(shuffled_model, codec, items),
            "verified": score(gold_model),
            "shuffled": score(shuffled_model),
            "gold_optimizer_curve": gold_curve,
            "shuffled_optimizer_curve": control_curve,
        }
        history.append(phase)
        print(json.dumps({
            "stage_complete": stage_number,
            "gold_train_accuracy": phase["gold_training_qa"]["accuracy"],
            "gold_legacy_accuracy": phase["verified"]["legacy_40"]["accuracy"],
            "gold_transfer_accuracy": phase["verified"]["transfer_40"]["accuracy"],
            "heldout_language_loss": phase["verified"]["heldout_token_loss"],
        }), flush=True)

    final_gold = score(gold_model)
    final_shuffled = score(shuffled_model)
    # A mirrored pair counts as one underlying question, not two independent
    # held-out examples. These thresholds are only a research filter.
    no_major_regression = final_gold["heldout_token_loss"] <= (
        frozen_scores["heldout_token_loss"] * 1.05)
    clear_transfer = all(final_gold[name]["correct"] >= 32 and
                         final_gold[name].get("both_mirrors_correct", 0) >= 12
                         for name in ("legacy_40", "transfer_40"))
    out_report = {
        "schema": 1, "status": "experimental_not_promoted",
        "source_training_run": 41572257363,
        "source_step": checkpoint["step"],
        "source_checkpoint_sha256": hashlib.sha256(
            (folder / "checkpoint.pt").read_bytes()).hexdigest(),
        "dataset_train_sha256": manifest["train_sha256"],
        "dataset_valid_sha256": manifest["valid_sha256"],
        "original_random_initialized_parameters": frozen.num_parameters,
        "seed": seed, "total_adaptation_updates_per_arm":
            sum(x[1] for x in stages),
        "frozen": frozen_scores,
        "stages": history,
        "final_gold": final_gold, "final_shuffled": final_shuffled,
        "gold_fixed_seed_generation": generate_samples(
            gold_model, codec, seed, new_tokens=64),
        "rehearsal_preserves_language_within_five_percent": no_major_regression,
        "passes_preliminary_mirrored_qa_filter": clear_transfer,
        "human_graded_free_form_instructions": False,
        "language_proficiency_certified": False,
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "limitations": [
            "Small deterministic teaching templates and one seed",
            "Train/test people disjoint, but tasks share abstract relations",
            "A/B options are not an open-ended dialogue or general intelligence",
            "Rehearsal only approximates retention of story generation",
            "Shuffled controls are not same-compute pure language continuations",
            "No model weights promoted to Android, public release or core assistant",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "staged-results.json").write_text(
        json.dumps(out_report, indent=2, ensure_ascii=False) + "\n")
    for name, model in (("gold", gold_model), ("shuffled", shuffled_model)):
        torch.save({
            "schema": 4, "experimental_only": True,
            "adaptation": "preschool_qa_staged",
            "arm": name, "configuration": asdict(model.cfg),
            "source_checkpoint_sha256": out_report["source_checkpoint_sha256"],
            "model": model.state_dict(),
            "updates": out_report["total_adaptation_updates_per_arm"],
        }, output / f"{name}-research.pt")
    print(json.dumps({
        "frozen": frozen_scores, "gold": final_gold,
        "shuffled": final_shuffled, "quality_filter": {
            "language_retained": no_major_regression,
            "qa_mirror_threshold": clear_transfer,
        }, "duration_sec": out_report["elapsed_seconds"],
    }, indent=2), flush=True)
    return out_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("artifacts/qa-staged"))
    p.add_argument("--seed", type=int, default=99)
    args = p.parse_args()
    curriculum(folder=args.checkpoint_dir, output=args.out, seed=args.seed)


if __name__ == "__main__":
    main()
