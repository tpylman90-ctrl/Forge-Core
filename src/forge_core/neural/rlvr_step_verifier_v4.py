"""Forge v4: balance step verification before attempting further RLVR.

A standalone scratch-started neural verifier is trained on BOTH steps of
each synthetic formal task, with canonical verified prefixes. Compare
ordinary BCE, positive-weighted BCE, and three-action oracle CE.

Process labels enter losses ONLY; policy.forward receives task features
and proposed states, never correctness or an "available" bit. Thresholds
are selected on development cases, not shifted final evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

from forge_core.neural.rlvr_abstention_v3 import (
    AbstentionPolicy, SEEDS, case_for, candidates, evaluate, gold_action,
)
from forge_core.neural.rlvr_process_v1 import encode, expected_step

ARMS = ("unweighted_validity_bce", "balanced_validity_bce", "three_action_ce")
THRESHOLDS = (.1, .2, .3, .4, .5, .6, .7, .8, .9)
TRAIN_EXAMPLES = 16


def canonical_step_examples(seed: int, index: int, split: str):
    """Even impossible episodes supply stage-two training examples.

    At evaluation, the actual learned policy still stops on rejection.
    Canonical prefix access is allowed in offline training ONLY, and
    must never be represented as an on-policy rollout.
    """
    case = case_for(seed, index, split)
    rng = random.Random(seed * 100999 + index * 107 + 31)
    examples = []
    state = case.task.initial
    for step in range(2):
        offered = candidates(case, step, state, rng)
        target, verified = gold_action(case.task, step, state, offered)
        batch = torch.stack([encode(case.task, step, state, v) for v in offered])
        examples.append((batch, target,
                         torch.tensor(verified, dtype=torch.float32)))
        # Offline teacher-forced prefix for full step-2 coverage.
        state = expected_step(case.task, step, state)
    return examples


@torch.no_grad()
def candidate_metrics(model: AbstentionPolicy, *, seed: int,
                      split: str, n: int = 240) -> dict:
    """Positive recall, specificity and Brier on ALL offered candidates."""
    if n < 8 or n % 8:
        raise ValueError("Require a balanced problem sample")
    model.eval()
    tp = tn = fp = fn = 0
    brier = 0.0
    offered_valid_steps = 0
    offered_none_steps = 0
    for i in range(n):
        for data, target, labels in canonical_step_examples(seed, i, split):
            _, logits = model(data)
            probabilities = torch.sigmoid(logits).tolist()
            offered_valid_steps += int(target != 2)
            offered_none_steps += int(target == 2)
            for p, y in zip(probabilities, labels.tolist()):
                brier += (p - y) ** 2
                guessed = p >= .5
                if y and guessed:
                    tp += 1
                elif not y and guessed:
                    fp += 1
                elif y:
                    fn += 1
                else:
                    tn += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    return {
        "candidate_count": tp + tn + fp + fn,
        "true_positive": tp, "false_positive": fp,
        "true_negative": tn, "false_negative": fn,
        "valid_steps_offered": offered_valid_steps,
        "no_valid_steps_offered": offered_none_steps,
        "precision": round(precision, 4),
        "valid_recall": round(recall, 4),
        "invalid_specificity": round(specificity, 4),
        "balanced_accuracy": round((recall + specificity) / 2, 4),
        "raw_accuracy": round((tp + tn) / (tp + tn + fp + fn), 4),
        "brier": round(brier / (tp + tn + fp + fn), 5),
        "always_invalid_accuracy": round((tn + fp) / (tp + tn + fp + fn), 4),
        "always_invalid_balanced_accuracy": .5,
    }


def balanced_gate(result: dict) -> tuple[float, float, float]:
    """Harmonic mean penalizes zero valid-proof coverage explicitly."""
    f = result["feasible_proof_rate"]
    s = result["infeasible_safe_rate"]
    harmonic = 0 if f + s == 0 else 2 * f * s / (f + s)
    return round(harmonic, 6), result["resolved_rate"], -result["accepted_invalid_rate"]


@torch.no_grad()
def select_dev_threshold(model: AbstentionPolicy, *, seed: int) -> tuple[float, dict]:
    comparisons = {}
    winner = None
    best = (-1., -1., -1.)
    for threshold in THRESHOLDS:
        metric = evaluate(
            model, seed=seed, split="development", n=160,
            mode="checker", threshold=threshold)
        comparisons[str(threshold)] = {
            "resolved": metric["resolved"],
            "completed_feasible": metric["correct_feasible"],
            "correct_infeasible": metric["correct_infeasible"],
            "accepted_invalid_rate": metric["accepted_invalid_rate"],
        }
        key = balanced_gate(metric)
        if key > best:
            best, winner = key, threshold
    return winner, {
        "criterion": "harmonic mean of feasible-proof and infeasible-safe rates",
        "chosen_threshold": winner, "development_grid": comparisons,
        "development_score": best[0],
    }


def fit(arm: str, *, seed: int, updates: int = 100,
        tasks_per_update: int = TRAIN_EXAMPLES):
    if arm not in ARMS or not 1 <= updates <= 200 or not 8 <= tasks_per_update <= 32:
        raise ValueError("Unregistered CPU or optimization budget")
    torch.manual_seed(seed)
    model = AbstentionPolicy()
    opt = torch.optim.AdamW(model.parameters(), lr=.002)
    curve = []
    # Process class prevalence is one valid per four offered candidates:
    # pos_weight=3 makes missed positive instances comparably important.
    pos_weight = torch.tensor([3.], dtype=torch.float32)
    action_weights = torch.tensor([2., 2., 1.], dtype=torch.float32)
    for update in range(1, updates + 1):
        model.train()
        losses = []
        for j in range(tasks_per_update):
            case_index = (update - 1) * tasks_per_update + j
            for x, target, labels in canonical_step_examples(
                    seed, case_index, "train"):
                actions, process = model(x)
                if arm == "three_action_ce":
                    loss = F.cross_entropy(
                        actions.unsqueeze(0),
                        torch.tensor([target], dtype=torch.long),
                        weight=action_weights)
                    loss = loss + .2 * F.binary_cross_entropy_with_logits(
                        process, labels, pos_weight=pos_weight)
                else:
                    loss = F.binary_cross_entropy_with_logits(
                        process, labels,
                        pos_weight=(pos_weight if arm == "balanced_validity_bce"
                                    else None))
                losses.append(loss)
        opt.zero_grad(set_to_none=True)
        loss = torch.stack(losses).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if update == 1 or update == updates or update % 20 == 0:
            curve.append({"update": update, "training_loss": round(float(loss.detach()), 5)})
    recovery = {
        "schema": 1, "research_only": True, "seed": seed,
        "arm": arm, "completed_updates": updates,
        "weights": model.state_dict(), "optimizer_state": opt.state_dict(),
        "torch_rng_state": torch.get_rng_state(), "training_curve": curve,
    }
    return model, curve, recovery


def experiment(out: Path, *, updates: int = 100,
               seeds: tuple[int, ...] = SEEDS) -> dict:
    if not 10 <= updates <= 200 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid step-verifier research configuration")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    out.mkdir(parents=True, exist_ok=True)
    final = []
    for seed in seeds:
        baseline = {
            "always_reject": evaluate(None, seed=seed+17201,
                                       split="shifted", n=240,
                                       mode="always_reject"),
            "exact_symbolic_oracle": evaluate(None, seed=seed+17201,
                                              split="shifted", n=240,
                                              mode="oracle"),
        }
        arms = {}
        for arm in ARMS:
            model, curve, state = fit(arm, seed=seed, updates=updates)
            torch.save(state, out / f"seed-{seed}-{arm}-research.pt")
            if arm == "three_action_ce":
                mode = "greedy"
                selected = {"criterion": "explicit 3-way actor action, no threshold"}
                cut = .5
            else:
                mode = "checker"
                cut, selected = select_dev_threshold(model, seed=seed+3997)
            arms[arm] = {
                "threshold_selection": selected,
                "dev_positive_and_negative": candidate_metrics(
                    model, seed=seed+3997, split="development"),
                "shifted_candidate_metrics": candidate_metrics(
                    model, seed=seed+17201, split="shifted"),
                "shifted_episode_metrics": evaluate(
                    model, seed=seed+17201, split="shifted", n=240,
                    mode=mode, threshold=cut),
                "iid_episode_metrics": evaluate(
                    model, seed=seed+8363, split="development",
                    mode=mode, threshold=cut),
                "training_curve": curve,
                "parameters": sum(p.numel() for p in model.parameters()),
            }
        final.append({"seed": seed, "references": baseline, "arms": arms})
        (out / f"seed-{seed}-partial.json").write_text(
            json.dumps(final[-1], indent=2) + "\n", encoding="utf-8")
    report = {
        "schema": 1, "experiment": "supervised-verified-steps-not-RLVR",
        "research_only": True, "promoted": False,
        "original_language_weights_unchanged": True,
        "training": "verified canonical-prefix states for both steps",
        "arms": list(ARMS),
        "optimization_updates_per_arm": updates,
        "train_seeds": list(seeds),
        "dev_thresholds_selected_without_shifted_test": True,
        "shifted_test_is_procedural_not_independently_authored": True,
        "comparison": "balanced step classification vs 3-action classifier",
        "results": final,
        "limitations": [
            "All predictors start randomly initialized; no pretrained model is used",
            "This is supervised step-validity pretraining, not claimed as new RLVR success",
            "Teacher-forced prefixes differ from actual stopped rollouts",
            "Original Forge Seed language weights and user memory are untouched",
            "A fixed synthetically generated truth oracle may have trivial shortcuts",
            "Perfect symbolic oracle baseline is not a learned benchmark result",
            "The same shifted rule families were previously studied: this is not a sealed final exam",
            "Balanced threshold selection uses development only; no outcome promotion",
        ],
    }
    (out / "rlvr-step-verifier-v4.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "updates": updates,
        "seeds": [{
            "seed": row["seed"],
            "learned": {
                arm: {
                    "resolved": data["shifted_episode_metrics"]["resolved"],
                    "feasible_proofs": data["shifted_episode_metrics"]["correct_feasible"],
                    "safe_rejections": data["shifted_episode_metrics"]["correct_infeasible"],
                    "valid_recall": data["shifted_candidate_metrics"]["valid_recall"],
                    "balanced_accuracy": data["shifted_candidate_metrics"]["balanced_accuracy"],
                } for arm, data in row["arms"].items()
            },
        } for row in final],
    }, indent=2), flush=True)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results/rlvr-step-v4"))
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--seeds", type=int, choices=(1, 2), default=2)
    args = p.parse_args()
    experiment(args.out, updates=args.updates, seeds=SEEDS[:args.seeds])


if __name__ == "__main__":
    main()
