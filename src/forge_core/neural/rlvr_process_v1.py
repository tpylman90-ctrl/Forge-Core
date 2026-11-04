"""Forge RLVR research: scratch-started policy for formally verifiable two-step proofs.

This is NOT fine-tuning the language model or free-form hidden CoT.
A small independent policy proposes explicit, inspectable intermediate
states. Deterministic arithmetic and propositional-rule oracles supply
outcome and process rewards. A process classifier learns from oracle
labels; policy gradients use only sampled rewards (no gold as inputs).

No exec/eval, user code, shell, outside model weights or private data.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

PROPOSITIONS = 6
FEATURE_DIM = 32
ARMS = ("outcome_only", "process_reward", "process_supervised")


@dataclass(frozen=True)
class Task:
    kind: str                    # "math" or "logic"
    initial: int                 # math value or set-of-propositions bitmask
    operations: tuple[int, int]  # math increments
    implications: tuple[tuple[int, int], tuple[int, int]]
    goal: int

    def __post_init__(self):
        if self.kind not in ("math", "logic"):
            raise ValueError("Unsupported proof domain")
        if self.kind == "logic":
            for source, target in self.implications:
                if not (0 <= source < PROPOSITIONS and
                        0 <= target < PROPOSITIONS and source != target):
                    raise ValueError("Invalid propositional implication")


def expected_step(task: Task, step: int, state: int) -> int:
    """Pure, independently verifiable single transition."""
    if step not in (0, 1):
        raise ValueError("Only two-step bounded proofs are supported")
    if task.kind == "math":
        return state + task.operations[step]
    antecedent, consequent = task.implications[step]
    return state | (1 << consequent) if state & (1 << antecedent) else state


def verify_step(task: Task, step: int, before: int, proposed: int) -> bool:
    return proposed == expected_step(task, step, before)


def verify_trace(task: Task, trace: tuple[int, int]) -> dict:
    """Reward the actual proposed steps, NOT an ideal reconstructed proof."""
    if len(trace) != 2:
        raise ValueError("Every trace must have two explicit steps")
    first_valid = verify_step(task, 0, task.initial, trace[0])
    second_valid = verify_step(task, 1, trace[0], trace[1])
    # Exact final state, not just a coincidentally true subexpression.
    final_valid = trace[1] == task.goal
    return {
        "process": [bool(first_valid), bool(second_valid)],
        "final": bool(final_valid),
        "complete_verified_proof": bool(first_valid and second_valid and final_valid),
    }


def generate_task(seed: int, index: int, split: str = "train") -> Task:
    """Disjoint held-out operation ranges, no user data."""
    if split not in ("train", "exam") or index < 0:
        raise ValueError("Unknown experiment split")
    rng = random.Random(seed * 1000003 + index * 19 + (0 if split == "train" else 872719))
    kind = ("math", "logic")[index % 2]
    if kind == "math":
        initial = rng.randint(-5, 5) if split == "train" else rng.randint(12, 25) * rng.choice((-1, 1))
        choices = list(range(-4, 0)) + list(range(1, 5)) if split == "train" else (
            list(range(-8, -4)) + list(range(5, 9)))
        delta_a, delta_b = rng.choice(choices), rng.choice(choices)
        return Task(kind, initial, (delta_a, delta_b), ((0, 1), (1, 2)),
                    initial + delta_a + delta_b)
    # Proof chaining: p implies q; q implies r. Fresh proposition sets on exam.
    source, middle, target = rng.sample(range(PROPOSITIONS), 3)
    initial = 1 << source
    if split == "exam":
        initial |= 1 << rng.choice([i for i in range(PROPOSITIONS)
                                    if i not in (source, middle, target)])
    pair = ((source, middle), (middle, target))
    result = expected_step(Task("logic", initial, (0, 0), pair, 0), 0, initial)
    goal = expected_step(Task("logic", initial, (0, 0), pair, 0), 1, result)
    return Task("logic", initial, (0, 0), pair, goal)


def options(task: Task, step: int, state: int, rng: random.Random) -> tuple[int, int]:
    correct = expected_step(task, step, state)
    if task.kind == "math":
        wrong = correct + rng.choice((-3, -2, -1, 1, 2, 3))
    else:
        # Corrupt exactly one proposition bit. The correct state is never
        # identified by its position, and no true label enters features.
        wrong = correct ^ (1 << rng.randrange(PROPOSITIONS))
    if correct == wrong:
        raise AssertionError("Distractor equals verified transition")
    pair = (correct, wrong)
    return pair if rng.randrange(2) else pair[::-1]


def encode(task: Task, step: int, state: int, proposed: int) -> torch.Tensor:
    """Inputs contain task and candidate, NEVER verifier verdicts."""
    values = [
        1.0 if task.kind == "math" else 0.0,
        1.0 if task.kind == "logic" else 0.0,
        float(step),
        task.initial / 32.0 if task.kind == "math" else 0.0,
        state / 32.0 if task.kind == "math" else 0.0,
        proposed / 32.0 if task.kind == "math" else 0.0,
        task.operations[step] / 8.0 if task.kind == "math" else 0.0,
        task.operations[1 - step] / 8.0 if task.kind == "math" else 0.0,
    ]
    if task.kind == "logic":
        before = [float(bool(state & (1 << i))) for i in range(PROPOSITIONS)]
        candidate = [float(bool(proposed & (1 << i))) for i in range(PROPOSITIONS)]
        antecedent, consequent = task.implications[step]
        ant = [float(i == antecedent) for i in range(PROPOSITIONS)]
        cons = [float(i == consequent) for i in range(PROPOSITIONS)]
        values += before + candidate + ant + cons
    else:
        values += [0.0] * (FEATURE_DIM - len(values))
    if len(values) != FEATURE_DIM:
        raise AssertionError("Unbalanced proof candidate feature dimension")
    return torch.tensor(values, dtype=torch.float32)


class VerifiablePolicy(nn.Module):
    """Small, original randomly initialized policy and process verifier."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(FEATURE_DIM, 64), nn.Tanh(),
                                     nn.Linear(64, 64), nn.Tanh())
        self.action = nn.Linear(64, 1)
        self.process = nn.Linear(64, 1)

    def forward(self, candidates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if candidates.shape != (2, FEATURE_DIM):
            raise ValueError("Each proof choice needs two candidate states")
        hidden = self.encoder(candidates)
        return self.action(hidden).flatten(), self.process(hidden).flatten()


def rollout(policy: VerifiablePolicy, task: Task, rng: random.Random,
            *, greedy: bool = False) -> dict:
    state = task.initial
    trace: list[int] = []
    log_probs = []
    verifier_terms = []
    labels: list[bool] = []
    for step in range(2):
        choices = options(task, step, state, rng)
        values = torch.stack([encode(task, step, state, c) for c in choices])
        logits, validity_logits = policy(values)
        distribution = torch.distributions.Categorical(logits=logits)
        action = logits.argmax() if greedy else distribution.sample()
        picked = choices[int(action)]
        trace.append(picked)
        log_probs.append(distribution.log_prob(action))
        oracle = [verify_step(task, step, state, c) for c in choices]
        labels.extend(oracle)
        verifier_terms.append(F.binary_cross_entropy_with_logits(
            validity_logits, torch.tensor(oracle, dtype=torch.float32)))
        state = picked
    verdict = verify_trace(task, (trace[0], trace[1]))
    return {
        "trace": tuple(trace), "verdict": verdict,
        "logp": torch.stack(log_probs).sum(),
        "process_loss": torch.stack(verifier_terms).mean(),
        "reward_outcome": float(verdict["final"]),
        "reward_process": float(verdict["final"]) +
                          0.25 * sum(verdict["process"]),
        "verifier_labels": labels,
    }


@torch.no_grad()
def evaluate(policy: VerifiablePolicy, *, seed: int = 8113,
             n: int = 200, split: str = "exam") -> dict:
    """Fixed held-out problem index and RNG, independent of training RNG."""
    if not 2 <= n <= 1000:
        raise ValueError("Invalid evaluation size")
    policy.eval()
    counts = {"all_steps_and_final": 0, "outcome_correct": 0,
              "locally_valid_steps": 0, "math_verified": 0, "logic_verified": 0}
    kinds = {"math": 0, "logic": 0}
    for i in range(n):
        task = generate_task(seed, i, split)
        rng = random.Random(seed + i * 37)
        example = rollout(policy, task, rng, greedy=True)
        verdict = example["verdict"]
        counts["all_steps_and_final"] += verdict["complete_verified_proof"]
        counts["outcome_correct"] += verdict["final"]
        counts["locally_valid_steps"] += sum(verdict["process"])
        counts[f"{task.kind}_verified"] += verdict["complete_verified_proof"]
        kinds[task.kind] += 1
    return {
        "total": n, **counts, "total_local_steps": n * 2,
        "kind_totals": kinds,
        "verified_rate": round(counts["all_steps_and_final"] / n, 4),
    }


def train(*, arm: str, steps: int, seed: int, examples_per_update: int = 12,
          rollouts_per_example: int = 4) -> tuple[VerifiablePolicy, list[dict]]:
    if arm not in ARMS or not 1 <= steps <= 300 or not 2 <= examples_per_update <= 32:
        raise ValueError("Unsupported bounded RLVR training")
    if rollouts_per_example not in (2, 4, 8):
        raise ValueError("Invalid grouped rollout budget")
    torch.manual_seed(seed)
    policy = VerifiablePolicy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.002)
    rng = random.Random(seed + 1008)
    curve: list[dict] = []
    # All arms train from the SAME seed, optimizer hyperparams, and tasks.
    for update in range(1, steps + 1):
        losses = []
        successes = 0
        for i in range(examples_per_update):
            task = generate_task(seed, (update - 1) * examples_per_update + i)
            results = [rollout(policy, task, rng) for _ in range(rollouts_per_example)]
            rewards = torch.tensor([
                result["reward_outcome"] if arm == "outcome_only"
                else result["reward_process"] for result in results
            ], dtype=torch.float32)
            baseline = rewards.mean()
            # Score-function estimator; group-relative baseline is detached.
            # Use variable outcomes to ensure a real stochastic policy signal.
            pg = torch.stack([
                -(result["logp"] * (reward - baseline))
                for result, reward in zip(results, rewards)
            ]).mean()
            if arm == "process_supervised":
                pg = pg + 0.25 * torch.stack(
                    [r["process_loss"] for r in results]).mean()
            losses.append(pg)
            successes += sum(int(r["verdict"]["complete_verified_proof"])
                             for r in results)
        optimizer.zero_grad(set_to_none=True)
        torch.stack(losses).mean().backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        if update in (1, steps) or update % 20 == 0:
            curve.append({
                "update": update,
                "sampled_valid_proofs": successes,
                "sampled_total": examples_per_update * rollouts_per_example,
                "heldout_exam": evaluate(policy, seed=10007, n=60),
            })
    return policy, curve


def experiment(out: Path, *, seed: int = 104, steps: int = 100) -> dict:
    if not 10 <= steps <= 300:
        raise ValueError("Pilot requires 10..300 training updates")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    # A deterministic baseline and one independent held-out arithmetic/logic
    # distribution. The public report does not establish broad reasoning.
    torch.manual_seed(seed)
    initial = VerifiablePolicy()
    initial_eval = evaluate(initial, seed=10007, n=200)
    results = {}
    for arm in ARMS:
        policy, curve = train(arm=arm, steps=steps, seed=seed)
        results[arm] = {
            "exam": evaluate(policy, seed=10007, n=200),
            "curve": curve, "parameter_count": sum(p.numel() for p in policy.parameters()),
        }
    report = {
        "schema": 1, "status": "research_only", "promoted": False,
        "original_seed_model_updated": False,
        "environment": "bounded-two-step-additive-and-implication-proofs",
        "verifier": "deterministic exact state transition and final state",
        "learning": "on-policy REINFORCE grouped baseline; optional process-level BCE",
        "dataset": "original deterministic synthetic problems, no private data",
        "seed": seed, "steps": steps,
        "initial": initial_eval, "arms": results,
        "limits": [
            "The neural policy is independent from Forge's frozen text Transformer",
            "Traces are explicit proof states, not a general LLM chain-of-thought",
            "A correct final state alone does not prove valid intermediate reasoning",
            "No unsafe arbitrary code execution, no external code interpreter",
            "Small binary-action synthetic tasks, not mathematical theorem proving",
            "One seed, same exam during curve reporting; results are diagnostics only",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "rlvr-process-v1.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "initial": initial_eval,
        "final": {arm: r["exam"] for arm, r in results.items()},
    }, indent=2), flush=True)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("artifacts/rlvr-process-v1"))
    p.add_argument("--seed", type=int, default=104)
    p.add_argument("--steps", type=int, default=100)
    a = p.parse_args()
    experiment(a.out, seed=a.seed, steps=a.steps)


if __name__ == "__main__":
    main()
