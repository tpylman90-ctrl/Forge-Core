"""Forge Seed research v5: generate formal proof actions, not offered answers.

This original randomly initialized, standalone policy chooses an operation
from a bounded action vocabulary. An independent deterministic arithmetic
or finite-Horn-logic interpreter validates the action and its resulting state.

Outcome RLVR, dense process RLVR and oracle-supervised proposal learning
receive matched update schedules; the supervised arm receives additional
ground-truth information and must NOT be presented as pure RL improvement.

No language-model weights, pretrained models, raw CoT, user data or shell.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import random
import resource
import time

import torch
from torch import nn
from torch.nn import functional as F

PROPS = 8
HORIZON = 3
ACTION_NAMES = ("ADD_ONE", "SUB_ONE", "DOUBLE", "HALVE_EXACT_OR_RULE_3", "ABSTAIN")
STOP = 4
FEATURES = 48
SEEDS = (5321, 5322)
ARMS = ("outcome_rlvr", "step_reward_rlvr", "oracle_process_supervised")
SPLITS = ("train", "development", "shifted")


@dataclass(frozen=True)
class Problem:
    domain: str
    initial: int
    target: int
    rules: tuple[tuple[int, int], ...]
    horizon: int = HORIZON

    def __post_init__(self):
        if self.domain not in ("arithmetic", "implication") or self.horizon != HORIZON:
            raise ValueError("Unsupported bounded formal domain")
        if self.domain == "implication":
            if len(self.rules) != 4 or not 0 <= self.target < PROPS:
                raise ValueError("Logic goal/rule shape invalid")
            for a, b in self.rules:
                if not 0 <= a < PROPS or not 0 <= b < PROPS or a == b:
                    raise ValueError("Invalid Horn implication endpoints")


def achieved(p: Problem, state: int) -> bool:
    return state == p.target if p.domain == "arithmetic" else bool(state & (1 << p.target))


def transition(p: Problem, state: int, action: int) -> int | None:
    """Pure formal interpreter. None means illegal; STOP has no transition."""
    if not 0 <= action < STOP:
        return None
    if p.domain == "arithmetic":
        if action == 0:
            value = state + 1
        elif action == 1:
            value = state - 1
        elif action == 2:
            value = state * 2
        else:
            if state % 2:
                return None
            value = state // 2
        return value if -64 <= value <= 64 else None
    a, b = p.rules[action]
    if not state & (1 << a) or state & (1 << b):
        return None
    return state | (1 << b)


@lru_cache(maxsize=100000)
def best_continuations(p: Problem, state: int, remaining: int) -> tuple[int, ...]:
    """Exhaustive exact bounded reachability; not a neural reward model.

    Returns ALL actions that preserve at least one valid proof within
    the residual horizon. STOP is correct only when no solution exists.
    """
    if not 0 <= remaining <= HORIZON:
        raise ValueError("Invalid bounded search depth")
    if achieved(p, state):
        return ()
    viable = []
    if remaining:
        for action in range(STOP):
            nxt = transition(p, state, action)
            if nxt is not None and (
                achieved(p, nxt) or
                (remaining > 1 and best_continuations(p, nxt, remaining - 1) != (STOP,))
            ):
                viable.append(action)
    return tuple(viable) if viable else (STOP,)


def oracle_trace(p: Problem) -> tuple[int, ...]:
    """Construct formal proof tokens using BFS-reachability, never candidate answers."""
    state = p.initial
    history = []
    for step in range(HORIZON):
        if achieved(p, state):
            return tuple(history)
        good = best_continuations(p, state, HORIZON - step)
        action = good[0]
        if action == STOP:
            return tuple(history + [STOP])
        state = transition(p, state, action)
        history.append(action)
    return tuple(history)


def verify_trace(p: Problem, proposed: tuple[int, ...]) -> dict:
    """Recheck ACTUALLY PROPOSED actions, not a reconstructed ideal proof."""
    if not isinstance(proposed, tuple) or len(proposed) > HORIZON:
        raise ValueError("Trace must be an explicit tuple within depth cap")
    state = p.initial
    illegal = 0
    stopped = False
    for i, action in enumerate(proposed):
        if not isinstance(action, int) or not 0 <= action <= STOP:
            raise ValueError("Unregistered action symbol")
        if achieved(p, state):
            illegal += 1
            break
        if action == STOP:
            stopped = True
            if i != len(proposed) - 1:
                illegal += 1
            break
        nxt = transition(p, state, action)
        if nxt is None:
            illegal += 1
            break
        state = nxt
    proof = illegal == 0 and achieved(p, state) and not stopped
    # A refusal is only justified if the original task is provably
    # unsolvable within the horizon. It cannot be a shortcut to success.
    can_solve = best_continuations(p, p.initial, HORIZON) != (STOP,)
    sound_refusal = illegal == 0 and stopped and not can_solve
    return {
        "proved": bool(proof),
        "sound_refusal": bool(sound_refusal),
        "correct_resolution": bool(proof or sound_refusal),
        "invalid_actions": illegal,
        "solvable": bool(can_solve),
        "final_state": state,
    }


def reachable_states(p: Problem) -> dict[int, int]:
    """Reference breadth-first enumeration to construct balanced tasks."""
    depths = {p.initial: 0}
    frontier = [p.initial]
    for depth in range(1, HORIZON + 1):
        next_frontier = []
        for current in frontier:
            for action in range(STOP):
                nxt = transition(p, current, action)
                if nxt is not None and nxt not in depths:
                    depths[nxt] = depth
                    next_frontier.append(nxt)
        frontier = next_frontier
    return depths


def make_problem(seed: int, index: int, split: str) -> Problem:
    if split not in SPLITS or index < 0:
        raise ValueError("Unknown task split")
    rng = random.Random(1000003 * seed + 7919 * index + {
        "train": 17, "development": 1087, "shifted": 9473}[split])
    domain = "arithmetic" if index % 2 == 0 else "implication"
    solvable = (index // 2) % 2 == 0
    if domain == "arithmetic":
        if split == "shifted":
            value = rng.choice((-1, 1)) * rng.randint(18, 28)
        else:
            value = rng.randint(-7, 7)
        blank = Problem(domain, value, 0, ())
        reached = reachable_states(blank)
        if solvable:
            choices = [goal for goal, d in reached.items() if 2 <= d <= 3]
        else:
            # Near reachable numeric values: goal range overlaps plausible
            # endpoints, but exact bounded search certifies impossibility.
            lo, hi = min(reached), max(reached)
            choices = [goal for goal in range(max(-64, lo - 3), min(64, hi + 3) + 1)
                       if goal not in reached]
        if not choices:
            raise ValueError("Failed to construct balanced bounded math task")
        task = Problem(domain, value, rng.choice(choices), ())
    else:
        root, a, b, tip, decoy_src, decoy_dst = rng.sample(range(PROPS), 6)
        rules = [(root, a), (a, b), (b, tip), (decoy_src, decoy_dst)]
        rng.shuffle(rules)
        initial = 1 << root
        if split == "shifted":
            unused = [x for x in range(PROPS)
                      if x not in (root, a, b, tip, decoy_src, decoy_dst)]
            initial |= 1 << rng.choice(unused)
        target = tip if solvable else decoy_dst
        task = Problem(domain, initial, target, tuple(rules))
    is_solvable = best_continuations(task, task.initial, HORIZON) != (STOP,)
    if is_solvable != solvable:
        raise ValueError("Synthetic task failed independent reachability check")
    return task


def features(p: Problem, state: int, remaining: int, action: int) -> torch.Tensor:
    """Only public task/state/action structure, no transition or correct label."""
    if not 0 <= action <= STOP or not 0 <= remaining <= HORIZON:
        raise ValueError("Unregistered action/step")
    base = [
        float(p.domain == "arithmetic"),
        float(p.domain == "implication"),
        remaining / HORIZON,
        state / 64.0 if p.domain == "arithmetic" else 0.0,
        p.target / 64.0 if p.domain == "arithmetic" else 0.0,
        (p.target - state) / 64.0 if p.domain == "arithmetic" else 0.0,
    ]
    base += [float(action == i) for i in range(STOP + 1)]
    base += ([float(bool(state & (1 << i))) for i in range(PROPS)]
             if p.domain == "implication" else [0.] * PROPS)
    base += ([float(i == p.target) for i in range(PROPS)]
             if p.domain == "implication" else [0.] * PROPS)
    if p.domain == "implication" and action != STOP:
        ant, cons = p.rules[action]
        base += [float(i == ant) for i in range(PROPS)]
        base += [float(i == cons) for i in range(PROPS)]
    else:
        base += [0.] * (2 * PROPS)
    if len(base) > FEATURES:
        raise AssertionError("Explicit feature budget exhausted")
    return torch.tensor(base + [0.] * (FEATURES - len(base)), dtype=torch.float32)


class ProposalPolicy(nn.Module):
    """One shared scorer proposes one of five formal action tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(FEATURES, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, p: Problem, state: int, remaining: int) -> torch.Tensor:
        inputs = torch.stack([
            features(p, state, remaining, i) for i in range(STOP + 1)])
        return self.score(inputs).flatten()


def rollout(policy: ProposalPolicy | None, p: Problem,
            rng: random.Random, *, mode: str = "sample") -> dict:
    if mode not in ("sample", "greedy", "random", "always_stop", "oracle"):
        raise ValueError("Unknown proof action mode")
    if mode in ("sample", "greedy") and policy is None:
        raise ValueError("Neural action mode requires a policy")
    state = p.initial
    trace = []
    logps = []
    oracle_losses = []
    good_steps = 0
    invalid = 0
    verifier_calls = 0
    for step in range(HORIZON):
        if achieved(p, state):
            break
        remaining = HORIZON - step
        correct = best_continuations(p, state, remaining)
        verifier_calls += 1
        if mode in ("sample", "greedy"):
            logits = policy(p, state, remaining)
            dist = torch.distributions.Categorical(logits=logits)
            action_tensor = dist.sample() if mode == "sample" else logits.argmax()
            action = int(action_tensor)
            if mode == "sample":
                logps.append(dist.log_prob(action_tensor))
                # A process label is a SET of valid continuations, not a
                # single arbitrary "best" answer. The model is rewarded
                # for any formal action on a successful residual proof.
                oracle_losses.append(-torch.logsumexp(
                    F.log_softmax(logits, dim=0)[list(correct)], dim=0))
        elif mode == "oracle":
            action = correct[0]
        elif mode == "always_stop":
            action = STOP
        else:
            action = rng.randrange(STOP + 1)
        trace.append(action)
        if action in correct:
            good_steps += 1
        if action == STOP:
            break
        nxt = transition(p, state, action)
        verifier_calls += 1
        if nxt is None:
            invalid += 1
            break
        state = nxt
    verdict = verify_trace(p, tuple(trace))
    if verdict["invalid_actions"] != invalid:
        raise AssertionError("Execution and independent trace verification disagree")
    return {
        "trace": tuple(trace),
        "verdict": verdict,
        "steps": len(trace),
        "good_steps": good_steps,
        "invalid_proposals": invalid,
        "verifier_calls": verifier_calls,
        "logp": torch.stack(logps).sum() if logps else None,
        "oracle_process_loss": (torch.stack(oracle_losses).mean()
                                if oracle_losses else None),
        "outcome_reward": float(verdict["correct_resolution"]),
        "dense_reward": (float(verdict["correct_resolution"])
                         + .25 * good_steps
                         - .15 * invalid),
    }


@torch.no_grad()
def evaluate(policy: ProposalPolicy | None, *, seed: int,
             split: str, n: int = 240, mode: str = "greedy") -> dict:
    if split not in SPLITS or n < 8 or n % 4 or n > 1000:
        raise ValueError("Expected balanced bounded evaluation")
    if policy is not None:
        policy.eval()
    total = {"tasks": n, "feasible": 0, "infeasible": 0,
             "proved": 0, "sound_refusal": 0, "correct_resolution": 0,
             "invalid_proposals": 0, "verifier_calls": 0, "actions_proposed": 0,
             "by_domain": {}}
    for i in range(n):
        p = make_problem(seed, i, split)
        result = rollout(policy, p, random.Random(seed * 109 + i * 77 + 3),
                         mode=mode)
        verdict = result["verdict"]
        k = p.domain
        subtotal = total["by_domain"].setdefault(
            k, {"tasks": 0, "proved": 0, "sound_refusal": 0})
        subtotal["tasks"] += 1
        subtotal["proved"] += int(verdict["proved"])
        subtotal["sound_refusal"] += int(verdict["sound_refusal"])
        total["feasible"] += int(verdict["solvable"])
        total["infeasible"] += int(not verdict["solvable"])
        for f in ("proved", "sound_refusal", "correct_resolution"):
            total[f] += int(verdict[f])
        total["invalid_proposals"] += result["invalid_proposals"]
        total["verifier_calls"] += result["verifier_calls"]
        total["actions_proposed"] += result["steps"]
    assert total["feasible"] == total["infeasible"] == n // 2
    total["mode"] = mode
    total["split"] = split
    total["resolution_rate"] = round(total["correct_resolution"] / n, 4)
    total["feasible_proof_rate"] = round(total["proved"] / total["feasible"], 4)
    total["infeasible_refusal_rate"] = round(total["sound_refusal"] / total["infeasible"], 4)
    return total


def balanced_gate(metric: dict) -> tuple[float, int]:
    """Always abstaining scores zero, no matter its total accuracy."""
    a, b = metric["feasible_proof_rate"], metric["infeasible_refusal_rate"]
    return (round(2 * a * b / (a + b), 6) if a + b else 0.0,
            metric["correct_resolution"])


def train_arm(arm: str, *, seed: int, updates: int = 100,
              tasks_per_update: int = 8,
              rollouts_per_task: int = 4,
              dev_every: int = 25) -> tuple[ProposalPolicy, dict]:
    if (arm not in ARMS or not 1 <= updates <= 250
            or not 2 <= tasks_per_update <= 16
            or rollouts_per_task not in (2, 4)
            or not 1 <= dev_every <= updates):
        raise ValueError("Outside bounded proof-training budget")
    torch.manual_seed(seed)
    policy = ProposalPolicy()
    opt = torch.optim.AdamW(policy.parameters(), lr=.0015)
    rng = random.Random(seed + 3293)
    best = None
    best_gate = (-1., -1)
    curve = []
    for update in range(1, updates + 1):
        policy.train()
        batch_losses = []
        for j in range(tasks_per_update):
            task = make_problem(
                seed, (update - 1) * tasks_per_update + j, "train")
            samples = [rollout(policy, task, rng) for _ in range(rollouts_per_task)]
            reward_key = ("outcome_reward" if arm == "outcome_rlvr"
                          else "dense_reward")
            rewards = torch.tensor([r[reward_key] for r in samples])
            relative_advantage = rewards - rewards.mean()
            terms = []
            for result, advantage in zip(samples, relative_advantage):
                term = -(result["logp"] * advantage)
                if arm == "oracle_process_supervised":
                    term = term + .5 * result["oracle_process_loss"]
                terms.append(term)
            batch_losses.append(torch.stack(terms).mean())
        opt.zero_grad(set_to_none=True)
        loss = torch.stack(batch_losses).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.)
        opt.step()
        if update == 1 or update % dev_every == 0 or update == updates:
            metric = evaluate(policy, seed=seed + 7211,
                              split="development", n=120)
            key = balanced_gate(metric)
            if key > best_gate:
                best_gate = key
                best = {
                    "step": update,
                    "weights": copy.deepcopy(policy.state_dict()),
                }
            curve.append({
                "update": update, "loss": round(float(loss.detach()), 5),
                "dev_correct": metric["correct_resolution"],
                "dev_proved": metric["proved"],
                "dev_refused": metric["sound_refusal"],
                "balanced_gate": key[0],
            })
            print(json.dumps({"arm": arm, "seed": seed, **curve[-1]}), flush=True)
    if best is None:
        raise RuntimeError("No development model checkpoint")
    last_state = copy.deepcopy(policy.state_dict())
    recovery = {
        "schema": 1, "research_only": True,
        "seed": seed, "arm": arm,
        "last_update": updates,
        "last_weights": last_state,
        "best_dev_step": best["step"],
        "best_weights": best["weights"],
        "optimizer_state": opt.state_dict(),
        "python_rng_state": rng.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "development_curve": curve,
    }
    policy.load_state_dict(best["weights"])
    return policy, recovery


def experiment(out: Path, *, updates: int = 100,
               seeds: tuple[int, ...] = SEEDS) -> dict:
    if not 10 <= updates <= 250 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Unsupported experiment budget/seeds")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    results = []
    for seed in seeds:
        torch.manual_seed(seed)
        untrained = ProposalPolicy()
        baseline = {
            "random_policy": evaluate(None, seed=seed + 13021,
                                      split="shifted", mode="random"),
            "untrained_neural": evaluate(untrained, seed=seed + 13021,
                                        split="shifted"),
            "always_stop": evaluate(None, seed=seed + 13021,
                                    split="shifted", mode="always_stop"),
            "exact_planner": evaluate(None, seed=seed + 13021,
                                      split="shifted", mode="oracle"),
        }
        arms = {}
        for arm in ARMS:
            model, recovery = train_arm(arm, seed=seed,
                                         updates=updates)
            torch.save(recovery, out / f"seed-{seed}-{arm}-recovery.pt")
            arms[arm] = {
                "best_development_step": recovery["best_dev_step"],
                "development_curve": recovery["development_curve"],
                "independent_iid": evaluate(
                    model, seed=seed + 8117, split="development"),
                "shifted_exam": evaluate(
                    model, seed=seed + 13021, split="shifted"),
                "parameters": sum(p.numel() for p in model.parameters()),
            }
        results.append({"seed": seed, "baselines": baseline, "arms": arms})
        (out / f"seed-{seed}-results.json").write_text(
            json.dumps(results[-1], indent=2) + "\n", encoding="utf-8")
    report = {
        "schema": 1, "study": "from-scratch-propose-and-exact-verify",
        "status": "research_only", "promoted": False,
        "original_forge_seed_language_weights_unchanged": True,
        "action_symbols": list(ACTION_NAMES),
        "formal_horizon": HORIZON,
        "training_updates_per_arm_seed": updates,
        "train_seeds": list(seeds),
        "experiment_seconds": round(time.monotonic() - t0, 2),
        "peak_linux_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "checkpoint_selection": "development only",
        "exam": "held-out procedural, arithmetic-range and logic-premise shifted",
        "arms": list(ARMS), "results": results,
        "limitations": [
            "The trained policy is a separate original model, not the Forge language Transformer",
            "Proof traces are explicit action symbols, not private model CoT",
            "An exact bounded search oracle generates process rewards and supervision",
            "Process-supervised training receives more verifier labels than pure RL",
            "This is a finite action vocabulary, not general theorem proving",
            "Shifted tasks are synthetic and not human-authored unseen mathematics",
            "No external execution, personal user data, Android code, or checkpoint promotion",
        ],
    }
    (out / "rlvr-proof-generator-v5.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "seconds": report["experiment_seconds"],
        "seeds": [{
            "seed": x["seed"],
            "baselines": {
                b: (v["proved"], v["sound_refusal"])
                for b, v in x["baselines"].items()},
            "learned": {
                a: (v["shifted_exam"]["proved"],
                    v["shifted_exam"]["sound_refusal"])
                for a, v in x["arms"].items()},
        } for x in results],
    }, indent=2), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results/rlvr-generator-v5"))
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--seeds", type=int, choices=(1, 2), default=2)
    a = p.parse_args()
    experiment(a.out, updates=a.updates, seeds=SEEDS[:a.seeds])


if __name__ == "__main__":
    main()
