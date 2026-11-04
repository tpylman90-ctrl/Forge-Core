"""Forge RLVR v3: learned verification with an explicit NONE VALID action.

Two candidate proof steps may include one valid state or NO valid state.
Reject is an independent third policy action. The pure formal oracle
supplies labels/rewards; no gold fields enter policy.forward().

This is a standalone original randomly initialized neural experiment.
It does not update Forge Seed language weights or expose private CoT.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

from forge_core.neural.rlvr_process_v1 import (
    PROPOSITIONS, FEATURE_DIM, Task, encode, expected_step, generate_task,
    verify_step, verify_trace,
)

REJECT = 2
ARMS = ("outcome_only", "dense_reward", "oracle_process_supervision")
SEEDS = (311, 719)
SPLITS = ("train", "development", "shifted")
# Deliberately one pair-valid task and three infeasible (some at step 0,
# some at step 1) per four examples, balanced independently of task kind.
REGIMES = ((True, True), (True, False), (False, True), (False, False))


@dataclass(frozen=True)
class Case:
    task: Task
    availability: tuple[bool, bool]
    domain: str
    split: str


def case_for(seed: int, index: int, split: str) -> Case:
    if split not in SPLITS or index < 0:
        raise ValueError("Unregistered verifier experiment split")
    # Train/dev share numeric families but have unrelated seeds and indices.
    # The shifted split contains larger arithmetic values and extra logic
    # premises. None reuse the prior experiment's published test seeds.
    source = "exam" if split == "shifted" else "train"
    salt = {"train": 1181, "development": 5303, "shifted": 9037}[split]
    task = generate_task(seed + salt, index + (100_000 if split != "train" else 0), source)
    availability = REGIMES[((index // 2) + seed + salt) % 4]
    return Case(task, availability, task.kind, split)


def candidates(case: Case, step: int, state: int,
               rng: random.Random) -> tuple[int, int]:
    """Generate unlabeled candidates with either one or zero valid choices."""
    correct = expected_step(case.task, step, state)
    if case.task.kind == "math":
        offsets = rng.sample((-4, -3, -2, -1, 1, 2, 3, 4), 2)
        wrongs = (correct + offsets[0], correct + offsets[1])
    else:
        bits = rng.sample(range(PROPOSITIONS), 2)
        wrongs = (correct ^ (1 << bits[0]), correct ^ (1 << bits[1]))
    offered = (correct, wrongs[0]) if case.availability[step] else wrongs
    if offered[0] == offered[1]:
        raise AssertionError("Identical offered proof states")
    if rng.randrange(2):
        offered = offered[::-1]
    verified = tuple(verify_step(case.task, step, state, x) for x in offered)
    if sum(verified) != int(case.availability[step]):
        raise AssertionError("Formal verifier disagrees with case availability")
    return offered


def gold_action(task: Task, step: int, state: int,
                offered: tuple[int, int]) -> tuple[int, tuple[bool, bool]]:
    """Only the independent exact checker may decide the correct action."""
    if len(offered) != 2 or offered[0] == offered[1]:
        raise ValueError("Expected two distinct candidate states")
    labels = tuple(verify_step(task, step, state, value) for value in offered)
    if sum(labels) > 1:
        raise ValueError("Ambiguous proof candidates")
    return (labels.index(True) if any(labels) else REJECT), labels


class AbstentionPolicy(nn.Module):
    """Original randomly initialized candidate scorer and reject head."""

    def __init__(self) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(FEATURE_DIM, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.pick = nn.Linear(64, 1)
        self.check = nn.Linear(64, 1)
        self.reject = nn.Sequential(
            nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1),
        )

    def forward(self, offered: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if offered.shape != (2, FEATURE_DIM):
            raise ValueError("Provide two unlabeled encoded proof candidates")
        hidden = self.feature(offered)
        # Reject score is permutation invariant over the two offered states.
        aggregate = torch.cat(
            (hidden.mean(0), hidden.abs().mean(0)), dim=0)
        action_logits = torch.cat(
            (self.pick(hidden).flatten(), self.reject(aggregate).flatten()))
        validity_logits = self.check(hidden).flatten()
        return action_logits, validity_logits


def rollout(policy: AbstentionPolicy | None, case: Case,
            rng: random.Random, *, mode: str = "sample") -> dict:
    """The verifier observes decisions but never supplies network inputs.

    Modes oracle and always_reject are explicit non-neural reference baselines.
    """
    if mode not in ("sample", "greedy", "always_reject", "oracle"):
        raise ValueError("Unknown policy selection mode")
    if mode in ("sample", "greedy") and policy is None:
        raise ValueError("Neural selection requires a policy")
    state = case.task.initial
    trace = []
    logps = []
    imitation = []
    process = []
    correct_moves = 0
    wrong_moves = 0
    invalid_accepted = 0
    false_reject = 0
    correct_reject = 0
    process_matches = 0
    brier = 0.0
    checked_candidates = 0
    accepted_actions = 0
    accepted_correct = 0
    for step in range(2):
        offered = candidates(case, step, state, rng)
        target, labels = gold_action(case.task, step, state, offered)
        if mode in ("sample", "greedy"):
            tensor = torch.stack([
                encode(case.task, step, state, v) for v in offered])
            logits, validity = policy(tensor)
            distribution = torch.distributions.Categorical(logits=logits)
            chosen = (int(distribution.sample()) if mode == "sample"
                      else int(logits.argmax()))
            if mode == "sample":
                logps.append(distribution.log_prob(
                    torch.tensor(chosen, dtype=torch.long)))
                imitation.append(F.cross_entropy(
                    logits.unsqueeze(0),
                    torch.tensor([target], dtype=torch.long)))
                process.append(F.binary_cross_entropy_with_logits(
                    validity, torch.tensor(labels, dtype=torch.float32)))
            probabilities = torch.sigmoid(validity.detach()).tolist()
            for label, probability in zip(labels, probabilities):
                process_matches += int((probability >= 0.5) == label)
                brier += (probability - float(label)) ** 2
                checked_candidates += 1
        else:
            chosen = target if mode == "oracle" else REJECT
        good = chosen == target
        correct_moves += int(good)
        wrong_moves += int(not good)
        if chosen == REJECT:
            correct_reject += int(good)
            false_reject += int(not good)
            break
        accepted_actions += 1
        if not labels[chosen]:
            invalid_accepted += 1
            break
        accepted_correct += 1
        state = offered[chosen]
        trace.append(state)

    complete_proof = False
    if len(trace) == 2:
        complete_proof = bool(verify_trace(case.task, tuple(trace))[
            "complete_verified_proof"])
    safe_rejection = correct_reject == 1 and invalid_accepted == 0
    resolved = bool(complete_proof or safe_rejection)
    return {
        "trace": tuple(trace), "resolved": resolved,
        "complete_proof": complete_proof, "safe_rejection": safe_rejection,
        "correct_moves": correct_moves, "wrong_moves": wrong_moves,
        "invalid_accepted": invalid_accepted, "false_reject": false_reject,
        "correct_reject": correct_reject,
        "accepted_actions": accepted_actions,
        "accepted_correct": accepted_correct,
        "logp": torch.stack(logps).sum() if logps else None,
        "actor_ce": torch.stack(imitation).mean() if imitation else None,
        "process_bce": torch.stack(process).mean() if process else None,
        "process_matches": process_matches,
        "brier": brier,
        "checked_candidates": checked_candidates,
        "terminal_reward": float(resolved),
        "dense_reward": float(resolved) + .3 * (correct_moves - wrong_moves),
    }


@torch.no_grad()
def evaluate(policy: AbstentionPolicy | None, *, seed: int,
             split: str, n: int = 240,
             mode: str = "greedy") -> dict:
    if split not in SPLITS or not 8 <= n <= 1000 or n % 8:
        raise ValueError("Evaluation needs balanced, bounded task count")
    if policy is not None:
        policy.eval()
    totals = defaultdict(int)
    by_domain = defaultdict(lambda: defaultdict(int))
    by_regime = defaultdict(lambda: defaultdict(int))
    for i in range(n):
        case = case_for(seed, i, split)
        rng = random.Random(seed * 9871 + i * 127 + 43)
        result = rollout(policy, case, rng, mode=mode)
        feasible = all(case.availability)
        fields = {
            "resolved": int(result["resolved"]),
            "complete_proof": int(result["complete_proof"]),
            "safe_rejection": int(result["safe_rejection"]),
            "feasible": int(feasible),
            "infeasible": int(not feasible),
            "correct_feasible": int(feasible and result["complete_proof"]),
            "correct_infeasible": int(not feasible and result["safe_rejection"]),
            "correct_reject": result["correct_reject"],
            "false_reject": result["false_reject"],
            "invalid_accepted": result["invalid_accepted"],
            "correct_moves": result["correct_moves"],
            "wrong_moves": result["wrong_moves"],
            "accepted_actions": result["accepted_actions"],
            "accepted_correct": result["accepted_correct"],
            "process_matches": result["process_matches"],
            "checked_candidates": result["checked_candidates"],
        }
        for k, v in fields.items():
            totals[k] += v
        for subset, key in ((by_domain, case.domain),
                            (by_regime, "".join("V" if v else "N" for v in case.availability))):
            subset[key]["tasks"] += 1
            subset[key]["resolved"] += int(result["resolved"])
            subset[key]["completed"] += int(result["complete_proof"])
        totals["brier_scaled"] += round(result["brier"] * 1_000_000)
    return {
        "total_tasks": n, "mode": mode, "split": split, **totals,
        "resolved_rate": round(totals["resolved"] / n, 4),
        "feasible_proof_rate": round(totals["correct_feasible"] /
                                     max(1, totals["feasible"]), 4),
        "infeasible_safe_rate": round(totals["correct_infeasible"] /
                                      max(1, totals["infeasible"]), 4),
        "accepted_invalid_rate": round(totals["invalid_accepted"] /
                                        max(1, totals["accepted_actions"]), 4),
        "validity_brier": round(totals["brier_scaled"] /
                                max(1, totals["checked_candidates"]) / 1_000_000, 5),
        "validity_accuracy": round(totals["process_matches"] /
                                   max(1, totals["checked_candidates"]), 4),
        "by_domain": {k: dict(v) for k, v in sorted(by_domain.items())},
        "by_offer_regime": {k: dict(v) for k, v in sorted(by_regime.items())},
    }


def fit_arm(arm: str, seed: int, *, updates: int = 80,
            tasks_per_update: int = 8, rollouts_per_task: int = 4,
            development_every: int = 20):
    if (arm not in ARMS or not 1 <= updates <= 220
            or not 2 <= tasks_per_update <= 16
            or rollouts_per_task not in (2, 4)
            or not 1 <= development_every <= updates):
        raise ValueError("Training outside reviewed CPU pilot limits")
    torch.manual_seed(seed)
    model = AbstentionPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015)
    rng = random.Random(seed + 7401)
    curve = []
    best_step = 0
    best_score = (-1, -1)
    best_state = None
    # Independent train seeds/cases are not used for development.
    for update in range(1, updates + 1):
        model.train()
        losses = []
        for task_no in range(tasks_per_update):
            case = case_for(seed, (update - 1) * tasks_per_update + task_no,
                            "train")
            rolls = [
                rollout(model, case, rng, mode="sample")
                for _ in range(rollouts_per_task)
            ]
            rewards = torch.tensor([
                r["terminal_reward"] if arm == "outcome_only" else r["dense_reward"]
                for r in rolls], dtype=torch.float32)
            # Grouped REINFORCE advantages, not PPO/GRPO.
            advantages = rewards - rewards.mean()
            per_roll = []
            for r, adv in zip(rolls, advantages):
                loss = -(r["logp"] * adv)
                if arm == "oracle_process_supervision":
                    loss = loss + .6 * r["actor_ce"] + .2 * r["process_bce"]
                per_roll.append(loss)
            losses.append(torch.stack(per_roll).mean())
        optimizer.zero_grad(set_to_none=True)
        torch.stack(losses).mean().backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update in (1, updates) or update % development_every == 0:
            result = evaluate(model, seed=seed+593, split="development",
                              n=80)
            key = (result["resolved"], result["correct_feasible"])
            if key > best_score:
                best_score = key
                best_step = update
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            curve.append({
                "update": update, "dev_resolved": result["resolved"],
                "dev_feasible_proofs": result["correct_feasible"],
                "dev_safe_rejections": result["correct_infeasible"],
            })
    model.load_state_dict(best_state)
    return model, curve, best_step


def experiment(out: Path, *, updates: int = 80,
               seeds: tuple[int, ...] = SEEDS) -> dict:
    if not 10 <= updates <= 220 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid research run configuration")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        initial = AbstentionPolicy()
        # Fixed seed and deterministic dev/shift test indices.
        torch.manual_seed(seed)
        initial = AbstentionPolicy()
        baselines = {
            "random_initial": evaluate(initial, seed=seed+11009,
                                       split="shifted"),
            "always_reject": evaluate(None, seed=seed+11009,
                                      split="shifted", mode="always_reject"),
            "exact_verifier": evaluate(None, seed=seed+11009,
                                       split="shifted", mode="oracle"),
        }
        results_for_seed = {}
        for arm in ARMS:
            model, curve, best_step = fit_arm(arm, seed, updates=updates,
                                             development_every=min(20, updates))
            results_for_seed[arm] = {
                "best_dev_update": best_step,
                "development_curve": curve,
                "iid_new_seed": evaluate(model, seed=seed+2117,
                                         split="development"),
                "shifted": evaluate(model, seed=seed+11009,
                                    split="shifted"),
                "num_parameters": sum(p.numel() for p in model.parameters()),
            }
        results.append({
            "seed": seed, "baselines": baselines, "arms": results_for_seed,
        })
        (out / f"seed-{seed}-partial.json").write_text(
            json.dumps(results[-1], indent=2) + "\n", encoding="utf-8")
    report = {
        "schema": 1, "research_only": True, "promoted": False,
        "canonical_forge_seed_unchanged": True,
        "task": "two-stage math/logic with 0-or-1 valid candidates plus explicit reject",
        "source": "original deterministic formal synthetic tasks",
        "oracle": "exact proof-step comparison and complete-proof checker",
        "train_updates_per_arm": updates,
        "training_seeds": list(seeds),
        "validation": "development checkpoint selection only",
        "evaluation": "new held-out problem seeds; distinct arithmetic ranges and logic premises",
        "note": "Same shifted generator family inspected in previous experiments, not an untouched external benchmark",
        "arms": list(ARMS), "results": results,
        "limits": [
            "The policy starts from random weights and is not Forge Seed's language model",
            "No candidate or gold validity label is supplied to network.forward()",
            "Process supervised arm receives MORE gold labels than RL-only controls",
            "Always-reject baseline is strong on infeasible tasks; evaluate feasible proofs separately",
            "No arbitrary code interpreter, free-form CoT, user memory, or external actions",
            "Neural correctness on small synthetic binary offers does not show real reasoning",
            "The exact symbolic verifier is a reference oracle, not a learned model",
        ],
    }
    (out / "rlvr-abstention-v3.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "updates": updates,
        "results": [{
            "seed": row["seed"],
            "always_reject": row["baselines"]["always_reject"]["resolved"],
            "oracle": row["baselines"]["exact_verifier"]["resolved"],
            "learned": {
                arm: {
                    "resolved": record["shifted"]["resolved"],
                    "feasible_proofs": record["shifted"]["correct_feasible"],
                    "safe_infeasible": record["shifted"]["correct_infeasible"],
                } for arm, record in row["arms"].items()
            },
        } for row in results],
    }, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/rlvr-abstention-v3"))
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--seeds", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    experiment(args.out, updates=args.updates, seeds=SEEDS[:args.seeds])


if __name__ == "__main__":
    main()
