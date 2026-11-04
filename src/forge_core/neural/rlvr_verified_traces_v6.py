"""Forge v6: process-supervised COMPLETE formal traces before RLVR.

Explicit proof-action policy, original random weights, exact symbolic oracle.
The teacher generates legal intermediate operations on CANONICAL solved
prefixes. No gold actions or teacher trace enter neural.forward().

Arms: outcome-only RLVR; verified trace imitation; imitation + on-policy
RLVR. Supervised arms have strictly more labels than the outcome baseline.
Never promote to canonical Forge language model or Android.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import resource
import time

import torch
from torch import nn
from torch.nn import functional as F

from forge_core.neural.rlvr_proof_generator_v5 import (
    HORIZON, SEEDS, STOP, ProposalPolicy, Problem, achieved, balanced_gate,
    best_continuations, evaluate, make_problem, oracle_trace, rollout,
    transition, verify_trace,
)

ARMS = ("outcome_rlvr", "verified_trace_imitation", "imitation_then_rlvr")


def proof_lessons(p: Problem) -> list[tuple[int, int, tuple[int, ...]]]:
    """Complete correct process trace, not the current weak policy's states."""
    oracle = oracle_trace(p)
    if not verify_trace(p, oracle)["correct_resolution"]:
        raise AssertionError("Independent formal oracle trace is not sound")
    state = p.initial
    output = []
    for step, action in enumerate(oracle):
        remaining = HORIZON - step
        allowed = best_continuations(p, state, remaining)
        if action not in allowed:
            raise AssertionError("Reference action leaves the successful proof set")
        output.append((state, remaining, allowed))
        if action == STOP:
            break
        next_state = transition(p, state, action)
        if next_state is None:
            raise AssertionError("Teacher proposed an invalid formal operation")
        state = next_state
    if not output:
        raise ValueError("Problem should not start in an already solved state")
    return output


def lesson_loss(policy: ProposalPolicy, p: Problem) -> torch.Tensor:
    """Set-based supervised loss: all valid continuations receive credit."""
    terms = []
    for state, remaining, acceptable in proof_lessons(p):
        logits = policy(p, state, remaining)
        logprob = F.log_softmax(logits, dim=0)
        terms.append(-torch.logsumexp(logprob[list(acceptable)], dim=0))
    return torch.stack(terms).mean()


@torch.no_grad()
def action_diagnostics(policy: ProposalPolicy, *, seed: int,
                       split: str, n: int = 120) -> dict:
    """Measure actions along fully verified teacher prefixes, no network labels."""
    if n < 8 or n % 4:
        raise ValueError("Use balanced diagnostic tasks")
    policy.eval()
    total = right = non_stop_total = non_stop_right = 0
    stop_total = stop_right = 0
    for index in range(n):
        p = make_problem(seed, index, split)
        for state, remaining, allowed in proof_lessons(p):
            action = int(policy(p, state, remaining).argmax())
            total += 1
            right += int(action in allowed)
            if allowed == (STOP,):
                stop_total += 1
                stop_right += int(action == STOP)
            else:
                non_stop_total += 1
                non_stop_right += int(action in allowed)
    return {
        "total_teacher_steps": total,
        "correct_teacher_steps": right,
        "valid_nonstop_actions": non_stop_right,
        "valid_nonstop_total": non_stop_total,
        "correct_abstain_actions": stop_right,
        "abstain_total": stop_total,
        "overall_step_accuracy": round(right / total, 4),
        "nonstop_accuracy": round(non_stop_right / max(1, non_stop_total), 4),
        "abstain_accuracy": round(stop_right / max(1, stop_total), 4),
    }


def policy_loss(policy: ProposalPolicy, p: Problem,
                rng: random.Random, n_rollouts: int = 4,
                *, reward_mode: str = "outcome") -> torch.Tensor:
    attempts = [rollout(policy, p, rng) for _ in range(n_rollouts)]
    key = "outcome_reward" if reward_mode == "outcome" else "dense_reward"
    rewards = torch.tensor([x[key] for x in attempts])
    advantages = rewards - rewards.mean()
    return torch.stack([
        -x["logp"] * advantage for x, advantage in zip(attempts, advantages)
    ]).mean()


def train_arm(arm: str, *, seed: int, updates: int = 600,
              tasks_per_update: int = 8,
              eval_every: int = 100) -> tuple[ProposalPolicy, dict]:
    if (arm not in ARMS or not 1 <= updates <= 1200
            or not 4 <= tasks_per_update <= 16
            or not 1 <= eval_every <= updates):
        raise ValueError("Outside approved trace-bootstrap budget")
    torch.manual_seed(seed)
    policy = ProposalPolicy()
    opt = torch.optim.AdamW(policy.parameters(), lr=.002)
    rng = random.Random(seed + 4481)
    best = None
    best_key = (-1., -1, -1.)
    curve = []
    warmup = max(10, updates // 4)
    for update in range(1, updates + 1):
        policy.train()
        losses = []
        for i in range(tasks_per_update):
            p = make_problem(seed, (update - 1) * tasks_per_update + i, "train")
            if arm == "outcome_rlvr":
                loss = policy_loss(policy, p, rng)
            elif arm == "verified_trace_imitation":
                loss = lesson_loss(policy, p)
            else:
                loss = lesson_loss(policy, p)
                if update > warmup:
                    loss = loss + .1 * policy_loss(
                        policy, p, rng, reward_mode="dense")
            losses.append(loss)
        opt.zero_grad(set_to_none=True)
        loss = torch.stack(losses).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.)
        opt.step()
        if update == 1 or update % eval_every == 0 or update == updates:
            dev = evaluate(policy, seed=seed+21721,
                           split="development", n=120)
            steps = action_diagnostics(policy, seed=seed+21721,
                                       split="development", n=120)
            gate = balanced_gate(dev)
            key = (gate[0], gate[1], steps["nonstop_accuracy"])
            if key > best_key:
                best_key = key
                best = {
                    "step": update,
                    "weights": copy.deepcopy(policy.state_dict()),
                }
            entry = {
                "update": update, "training_loss": round(float(loss.detach()), 5),
                "dev_proved": dev["proved"],
                "dev_sound_refusal": dev["sound_refusal"],
                "dev_balanced_gate": gate[0],
                "dev_nonstop_step_accuracy": steps["nonstop_accuracy"],
            }
            curve.append(entry)
            print(json.dumps({"seed": seed, "arm": arm, **entry}), flush=True)
    if best is None:
        raise RuntimeError("Missing valid development checkpoint")
    last = copy.deepcopy(policy.state_dict())
    recovery = {
        "schema": 1, "research_only": True,
        "arm": arm, "seed": seed, "updates": updates,
        "best_dev_step": best["step"],
        "last_weights": last,
        "best_weights": best["weights"],
        "optimizer_state": opt.state_dict(),
        "sampler_rng_state": rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "development_curve": curve,
    }
    policy.load_state_dict(best["weights"])
    return policy, recovery


def experiment(out: Path, *, updates: int = 600,
               seeds: tuple[int, ...] = SEEDS) -> dict:
    if not 10 <= updates <= 1200 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid research CPU budget or random seeds")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    out.mkdir(parents=True, exist_ok=True)
    begun = time.monotonic()
    result_rows = []
    for seed in seeds:
        baseline = {
            "always_stop": evaluate(
                None, seed=seed+31213, split="shifted", mode="always_stop"),
            "exact_oracle": evaluate(
                None, seed=seed+31213, split="shifted", mode="oracle"),
        }
        arms = {}
        for arm in ARMS:
            model, recovery = train_arm(
                arm, seed=seed, updates=updates,
                eval_every=min(100, updates))
            torch.save(recovery, out / f"seed-{seed}-{arm}-research.pt")
            arms[arm] = {
                "selected_dev_update": recovery["best_dev_step"],
                "development_curve": recovery["development_curve"],
                "development_proofs": evaluate(
                    model, seed=seed+21721, split="development", n=120),
                "development_teacher_steps": action_diagnostics(
                    model, seed=seed+21721, split="development", n=120),
                "shifted_proofs": evaluate(
                    model, seed=seed+31213, split="shifted"),
                "shifted_teacher_steps": action_diagnostics(
                    model, seed=seed+31213, split="shifted"),
                "parameters": sum(p.numel() for p in model.parameters()),
            }
        result_rows.append({
            "seed": seed, "baselines": baseline, "arms": arms})
        (out / f"seed-{seed}-results.json").write_text(
            json.dumps(result_rows[-1], indent=2) + "\n", encoding="utf-8")
    report = {
        "schema": 1, "experiment": "verified-trace-process-bootstrap",
        "research_only": True, "promoted": False,
        "original_forge_seed_language_checkpoint_unchanged": True,
        "train_updates_per_arm_seed": updates,
        "train_seeds": list(seeds),
        "experiment_seconds": round(time.monotonic()-begun, 2),
        "peak_linux_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "checkpoint_selection": "development only",
        "arm_names": list(ARMS), "results": result_rows,
        "limitations": [
            "Imitation and hybrid arms receive more exact-oracle labels than pure outcome RLVR",
            "Teaching uses complete reference traces, not model-discovered reasoning",
            "Action space and horizon are small and formal, not natural language or LLM CoT",
            "The procedural shifted family has been studied and is not externally certified",
            "No pretrained AI weights, arbitrary code evaluation, Android or private memory",
            "All policy weights are research-only; original Forge language model is unchanged",
        ],
    }
    (out / "rlvr-verified-traces-v6.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "seconds": report["experiment_seconds"],
        "results": [{
            "seed": r["seed"],
            "arms": {
                k: {
                    "proved": v["shifted_proofs"]["proved"],
                    "sound_refusal": v["shifted_proofs"]["sound_refusal"],
                    "nonstop_action_accuracy": v["shifted_teacher_steps"]["nonstop_accuracy"],
                } for k, v in r["arms"].items()
            },
        } for r in result_rows],
    }, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/rlvr-verified-traces-v6"))
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--seeds", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    experiment(args.out, updates=args.updates, seeds=SEEDS[:args.seeds])


if __name__ == "__main__":
    main()
