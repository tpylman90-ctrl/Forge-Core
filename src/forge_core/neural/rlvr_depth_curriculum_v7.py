"""Forge RLVR v7: learn formally verified proof actions with increasing depth.

Research-only scratch-started policy, no pretrained teacher weights.
The exact bounded formal planner generates supervision at depth 1, 2,
then 3, while impossible goals remain represented in each training phase.

Compare equal-budget flat long-proof imitation with staged imitation
and staged imitation plus a small genuine on-policy RLVR term.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import random
import resource
import time

import torch
from torch import nn
from torch.nn import functional as F

from forge_core.neural.rlvr_proof_generator_v5 import (
    HORIZON, STOP, ProposalPolicy, Problem, achieved,
    make_problem, reachable_states, rollout, transition, verify_trace,
)
from forge_core.neural.rlvr_verified_traces_v6 import policy_loss

ARMS = ("flat_long_proofs", "staged_proof_imitation", "staged_then_rlvr")
SEEDS = (5321, 5322)


def proof_depth(p: Problem) -> int | None:
    """Minimum formal action count to satisfy a goal, or None if impossible."""
    reach = reachable_states(p)
    if p.domain == "arithmetic":
        return reach.get(p.target)
    lengths = [depth for state, depth in reach.items()
               if state & (1 << p.target)]
    return min(lengths) if lengths else None


def problem_at_depth(seed: int, index: int, split: str,
                     depth: int) -> Problem:
    """Construct a NEW goal of exact optimal proof depth 1, 2 or 3.

    Never change the original task generator, old experiments or
    held-out results. All changed goals are checked by exact BFS.
    """
    if depth not in (1, 2, 3):
        raise ValueError("Only 1–3 step curriculum goals are supported")
    base = make_problem(seed, index, split)
    original = proof_depth(base)
    if original is None:
        return base
    reach = reachable_states(base)
    if base.domain == "arithmetic":
        choices = [state for state, d in reach.items() if d == depth]
    else:
        choices = []
        for atom in range(8):
            depths = [d for state, d in reach.items()
                      if state & (1 << atom)]
            if depths and min(depths) == depth:
                choices.append(atom)
    if not choices:
        raise ValueError("Unable to sample a reachable goal of requested depth")
    rng = random.Random(8111 * seed + 32771 * index + 13 * depth +
                        {"train": 1, "development": 2, "shifted": 3}[split])
    candidate = replace(base, target=rng.choice(choices))
    if proof_depth(candidate) != depth:
        raise AssertionError("Generated curriculum goal violates exact depth")
    if proof_depth(candidate) is None:
        raise AssertionError("Curriculum task marked feasible but has no proof")
    return candidate


def shortest_prefixes(p: Problem) -> list[tuple[int, int, tuple[int, ...]]]:
    """Formally verify a SHORTEST teacher trace and all shortest actions.

    Merely allowing a solution inside the three-step horizon would be
    incorrect for curriculum labels: a one-step task could be taught
    using a three-step detour.
    """
    distance = proof_depth(p)
    if distance is None:
        if not verify_trace(p, (STOP,))["sound_refusal"]:
            raise AssertionError("Impossible goal failed proof oracle")
        return [(p.initial, HORIZON, (STOP,))]
    if distance == 0:
        raise ValueError("Initial state is already solved")
    state = p.initial
    examples = []
    path = []
    for step in range(distance):
        actions = []
        for action in range(STOP):
            following = transition(p, state, action)
            if following is not None:
                remainder = proof_depth(replace(p, initial=following))
                if remainder == distance - step - 1:
                    actions.append(action)
        if not actions:
            raise AssertionError("Exact BFS found no shortest next operation")
        examples.append((state, HORIZON - step, tuple(actions)))
        chosen = actions[0]
        path.append(chosen)
        state = transition(p, state, chosen)
    if not achieved(p, state) or not verify_trace(p, tuple(path))["proved"]:
        raise AssertionError("Canonical shortest trace does not prove the goal")
    return examples


def shortest_lesson_loss(policy: ProposalPolicy, p: Problem) -> torch.Tensor:
    losses = []
    for state, remaining, allowed in shortest_prefixes(p):
        scores = F.log_softmax(policy(p, state, remaining), dim=0)
        losses.append(-torch.logsumexp(scores[list(allowed)], dim=0))
    return torch.stack(losses).mean()


def staged_task(seed: int, index: int, update: int,
                total_updates: int) -> tuple[Problem, int]:
    if not 1 <= update <= total_updates:
        raise ValueError("Unregistered proof curriculum update")
    # Phase I: only 1-step goals. Phase II: 1 and 2-step.
    # Phase III: a retained mix of all three difficulties.
    phase = min(3, 1 + 3 * (update - 1) // total_updates)
    requested_depth = 1 + ((index + seed) % phase)
    return problem_at_depth(seed, index, "train", requested_depth), phase


@torch.no_grad()
def evaluate_depth(policy: ProposalPolicy, *, seed: int,
                   split: str, depth: int,
                   n: int = 120) -> dict:
    if n < 8 or n % 4:
        raise ValueError("Depth test requires balanced task samples")
    policy.eval()
    proved = refused = illegal = 0
    good_nonstop = nonstop_total = 0
    for i in range(n):
        task = problem_at_depth(seed, i, split, depth)
        rng = random.Random(seed * 593 + i * 43)
        output = rollout(policy, task, rng, mode="greedy")
        proved += int(output["verdict"]["proved"])
        refused += int(output["verdict"]["sound_refusal"])
        illegal += output["invalid_proposals"]
        for state, remaining, acceptable in shortest_prefixes(task):
            if acceptable != (STOP,):
                nonstop_total += 1
                good_nonstop += int(
                    int(policy(task, state, remaining).argmax()) in acceptable)
    return {
        "total": n, "feasible": n // 2, "infeasible": n // 2,
        "proved": proved, "sound_refusal": refused,
        "correct_resolution": proved + refused,
        "illegal_proposals": illegal,
        "nonstop_correct": good_nonstop, "nonstop_total": nonstop_total,
        "nonstop_accuracy": round(good_nonstop / max(1, nonstop_total), 4),
        "feasible_rate": round(proved / (n // 2), 4),
        "infeasible_rate": round(refused / (n // 2), 4),
        "depth": depth, "split": split,
    }


def train_arm(arm: str, *, seed: int,
              updates: int = 900,
              tasks_per_update: int = 8,
              eval_every: int = 150):
    if (arm not in ARMS or not 1 <= updates <= 1500
            or not 4 <= tasks_per_update <= 16
            or not 1 <= eval_every <= updates):
        raise ValueError("Outside bounded v7 learning curriculum")
    torch.manual_seed(seed)
    policy = ProposalPolicy()
    opt = torch.optim.AdamW(policy.parameters(), lr=.002)
    rng = random.Random(seed + 9151)
    best = None
    best_key = (-1, -1, -1, -1.)
    curve = []
    for update in range(1, updates + 1):
        policy.train()
        losses = []
        for j in range(tasks_per_update):
            index = (update - 1) * tasks_per_update + j
            if arm == "flat_long_proofs":
                p = make_problem(seed, index, "train")
            else:
                p, phase = staged_task(seed, index, update, updates)
            loss = shortest_lesson_loss(policy, p)
            if arm == "staged_then_rlvr" and update > updates // 3:
                loss = loss + .1 * policy_loss(
                    policy, p, rng, reward_mode="dense")
            losses.append(loss)
        opt.zero_grad(set_to_none=True)
        loss = torch.stack(losses).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.)
        opt.step()
        if update == 1 or update % eval_every == 0 or update == updates:
            # The untouched shifted split is NEVER used for selecting weights.
            dev = {
                depth: evaluate_depth(policy, seed=seed + 3079,
                                      split="development", depth=depth)
                for depth in (1, 2, 3)
            }
            key = (
                dev[3]["proved"], dev[2]["proved"],
                dev[1]["proved"], dev[3]["nonstop_accuracy"])
            if key > best_key:
                best_key = key
                best = {
                    "step": update,
                    "weights": copy.deepcopy(policy.state_dict()),
                }
            entry = {
                "update": update,
                "loss": round(float(loss.detach()), 5),
                "development_completed_proofs_by_depth": {
                    str(k): dev[k]["proved"] for k in (1, 2, 3)},
                "development_hard_nonstop_accuracy": dev[3]["nonstop_accuracy"],
                "development_hard_safe_refusal": dev[3]["sound_refusal"],
            }
            curve.append(entry)
            print(json.dumps({"seed": seed, "arm": arm, **entry}), flush=True)
    if best is None:
        raise RuntimeError("Failed to select a development checkpoint")
    recovery = {
        "schema": 1, "research_only": True,
        "arm": arm, "seed": seed, "updates": updates,
        "best_step": best["step"],
        "best_weights": best["weights"],
        "last_weights": copy.deepcopy(policy.state_dict()),
        "optimizer_state": opt.state_dict(),
        "sampler_rng_state": rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "curve": curve,
    }
    policy.load_state_dict(best["weights"])
    return policy, recovery


def experiment(out: Path, *, updates: int = 900,
               seeds: tuple[int, ...] = SEEDS) -> dict:
    if not 10 <= updates <= 1500 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid depth curriculum study configuration")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    reports = []
    for seed in seeds:
        arms = {}
        for arm in ARMS:
            model, recovery = train_arm(arm, seed=seed, updates=updates,
                                         eval_every=min(150, updates))
            torch.save(recovery, out / f"seed-{seed}-{arm}-research.pt")
            arms[arm] = {
                "best_development_update": recovery["best_step"],
                "development_curve": recovery["curve"],
                "iid_development_by_depth": {
                    str(depth): evaluate_depth(
                        model, seed=seed+3079,
                        split="development", depth=depth)
                    for depth in (1, 2, 3)
                },
                "shifted_final_by_depth": {
                    str(depth): evaluate_depth(
                        model, seed=seed+13111,
                        split="shifted", depth=depth, n=240)
                    for depth in (1, 2, 3)
                },
            }
        reports.append({"seed": seed, "arms": arms})
        (out / f"seed-{seed}-results.json").write_text(
            json.dumps(reports[-1], indent=2) + "\n", encoding="utf-8")
    output = {
        "schema": 1, "study": "curriculum-formal-action-proof-depth",
        "research_only": True, "promoted": False,
        "original_forge_seed_language_checkpoint_unchanged": True,
        "scratch_started_new_policies": True,
        "updates_per_arm_seed": updates,
        "seeds": list(seeds),
        "experiment_seconds": round(time.monotonic() - started, 2),
        "peak_resident_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "selection": "development 3-step, then 2-step and 1-step proofs only",
        "shifted_procedural_exam_not_externally_sealed": True,
        "arms": list(ARMS), "results": reports,
        "limitations": [
            "Both lesson-based arms use verified oracle labels and are NOT pure RLVR",
            "Flat and curriculum arms receive different task difficulty sequences",
            "Shifted tasks use the same formal languages and have been studied before",
            "Neural proposed actions do not equal general causal or language reasoning",
            "No pretrained weights, Android modifications, external execution or private memory",
            "No checkpoint promotion; canonical scratch language training remains isolated",
        ],
    }
    (out / "rlvr-depth-curriculum-v7.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "seconds": output["experiment_seconds"],
        "shifted_proofs": [{
            "seed": row["seed"],
            "arms": {
                name: {
                    depth: value["proved"]
                    for depth, value in item["shifted_final_by_depth"].items()
                } for name, item in row["arms"].items()
            },
        } for row in reports],
    }, indent=2), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/rlvr-depth-v7"))
    parser.add_argument("--updates", type=int, default=900)
    parser.add_argument("--seeds", type=int, choices=(1, 2), default=2)
    a = parser.parse_args()
    experiment(a.out, updates=a.updates, seeds=SEEDS[:a.seeds])


if __name__ == "__main__":
    main()
