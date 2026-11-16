"""Three-lane progressive micro-curriculum with frequent retention checks.

Each lane and seed gets an isolated candidate. Language continues the verified
Forge parent. Blender/Godot use separate full-weight branches with 50% TinyStories
replay in every batch. Stages expand the training token prefix 25% -> 60% -> 100%.
The same held-out splits remain fixed across all stages and seeds.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from forge_core.neural.multilane_v2 import (
    BASE_RUN, BASE_SHA256, BASE_STEP, BASE_TOKENIZER_SHA256,
    BATCH, SEQ_LEN, EVAL_WINDOWS, TASK_PROMPTS, generate_task_samples,
    load_language_data, load_manual_data, sha256_file,
)
from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.seed_v07 import heldout
from forge_core.neural.train import make_batch

STAGES = (("foundation", 0.25), ("expansion", 0.60), ("full_mix", 1.0))
STAGE_UPDATES = 500
EVAL_EVERY = 100
CORE_REGRESSION_LIMIT = 0.05
CORE_REGRESSION_PATIENCE = 2
SEEDS = (43, 53, 73)


def digest_ids(ids: torch.Tensor) -> str:
    raw = ids.detach().to("cpu").contiguous().view(torch.uint8)
    digest = hashlib.sha256()
    chunk_size = 1 << 20
    for offset in range(0, raw.numel(), chunk_size):
        digest.update(bytes(raw[offset:offset + chunk_size].tolist()))
    return digest.hexdigest()


def progressive_pool(ids: torch.Tensor, fraction: float) -> torch.Tensor:
    """Use a growing, deterministic prefix of the pinned training content."""
    end = max(SEQ_LEN + 2, int(ids.numel() * fraction))
    end = min(ids.numel(), end)
    return ids[:end].contiguous()


def batch_for_lane(lane: str, domain_ids: torch.Tensor, core_ids: torch.Tensor,
                   rng: torch.Generator, replay_fraction: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    if lane == "language":
        return make_batch(domain_ids, BATCH, SEQ_LEN, rng)
    # Mix lane-specific manual tokens with the pinned TinyStories retention corpus.
    domain_count = min(BATCH - 1, max(1, round(BATCH * (1.0 - replay_fraction))))
    core_count = BATCH - domain_count
    dx, dy = make_batch(domain_ids, domain_count, SEQ_LEN, rng)
    cx, cy = make_batch(core_ids, core_count, SEQ_LEN, rng)
    return torch.cat((dx, cx), dim=0), torch.cat((dy, cy), dim=0)


def write_checkpoint(out: Path, lane: str, parent: dict, parent_sha: str,
                     parent_run: str, parent_step: int,
                     model: ByteTransformer, optimizer: torch.optim.Optimizer,
                     rng: torch.Generator, seed: int, updates: int,
                     best_step: int, best_loss: float, manifest: dict) -> None:
    ckpt = {
        "schema": 3,
        "lane": lane,
        "parent_run": parent_run,
        "parent_step": parent_step,
        "parent_sha256": parent_sha,
        "step": best_step,
        "lane_updates": int(parent.get("lane_updates", 0)) + updates,
        "updates_this_run": updates,
        "seed": seed,
        "configuration": model.cfg.__dict__,
        "revision": parent["revision"],
        "tokenizer_sha256": BASE_TOKENIZER_SHA256,
        "train_sha256": manifest["train_sha256"],
        "valid_sha256": manifest["valid_sha256"],
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "sample_rng_state": rng.get_state(),
        "source_manifest": manifest,
        "best_valid_loss": best_loss,
    }
    tmp = out / "checkpoint.pt.tmp"
    torch.save(ckpt, tmp)
    tmp.replace(out / "checkpoint.pt")


def run(lane: str, source: Path, out: Path, seed: int,
        stage_updates: int = STAGE_UPDATES, eval_every: int = EVAL_EVERY,
        parent_sha256: str | None = None, parent_run: str | None = None,
        expected_parent_seed: int | None = None, replay_fraction: float = 0.5,
        learning_rate_scale: float = 1.0) -> dict:
    if lane not in {"language", "blender", "godot"}:
        raise ValueError("Unknown lane")
    if seed not in SEEDS:
        raise ValueError(f"Seed must be one of {SEEDS}")
    if not 100 <= stage_updates <= 2000 or not 25 <= eval_every <= 500:
        raise ValueError("Invalid bounded micro-run configuration")
    if not 0.1 <= replay_fraction <= 0.9:
        raise ValueError("replay_fraction must be between 0.1 and 0.9")
    if not 0.25 <= learning_rate_scale <= 2.0:
        raise ValueError("learning_rate_scale must be between 0.25 and 2.0")

    parent_path = source / "checkpoint.pt"
    parent_sha = sha256_file(parent_path)
    expected_sha = parent_sha256 or BASE_SHA256
    if parent_sha != expected_sha:
        raise ValueError(f"Parent SHA256 mismatch: expected {expected_sha}, got {parent_sha}")
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    parent_step = int(parent.get("step", -1))
    parent_seed = int(parent.get("seed", -1))
    if parent_step < BASE_STEP or parent_seed not in SEEDS:
        raise ValueError("Parent checkpoint is not a recognized Forge Seed lineage")
    if parent_sha256 is None and (parent_step != BASE_STEP or parent_seed != 43):
        raise ValueError("Default parent must be the verified Forge 107250 seed-43 checkpoint")
    if expected_parent_seed is not None and parent_seed != expected_parent_seed:
        raise ValueError(f"Expected parent seed {expected_parent_seed}, got {parent_seed}")
    parent_run = parent_run or BASE_RUN
    tokenizer_path = source / "tokenizer.json"
    if sha256_file(tokenizer_path) != BASE_TOKENIZER_SHA256:
        raise ValueError("Tokenizer SHA256 mismatch")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    if lane == "language":
        _, train_ids, valid_ids, test_ids, manifest = load_language_data(source)
        core_train_ids, core_valid_ids = train_ids, valid_ids
        eval_windows = 2048
    else:
        train_ids, valid_ids, test_ids, manifest = load_manual_data(
            lane, source / "manual", tokenizer
        )
        core_train_ids = torch.tensor(
            tokenizer.encode((source / "train.txt").read_text(encoding="utf-8")).ids,
            dtype=torch.long,
        )
        core_valid_ids = torch.tensor(
            tokenizer.encode((source / "valid.txt").read_text(encoding="utf-8")).ids,
            dtype=torch.long,
        )
        eval_windows = EVAL_WINDOWS

    torch.set_num_threads(min(2, torch.get_num_threads()))
    torch.manual_seed(seed)
    model = ByteTransformer(ModelConfig(**parent["configuration"]))
    model.load_state_dict(parent["model"], strict=True)
    language_lr = float(parent.get("lr", 1.25e-5))
    learning_rate = (language_lr if lane == "language" else 1.0e-5) * learning_rate_scale
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    resume_branch = parent.get("lane") == lane and parent.get("optimizer") is not None
    resume_optimizer = lane == "language" or resume_branch
    if resume_optimizer:
        optimizer.load_state_dict(parent["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        if parent.get("torch_rng_state") is not None:
            torch.set_rng_state(parent["torch_rng_state"])
    rng = torch.Generator().manual_seed(seed + 1)
    if resume_optimizer and parent.get("sample_rng_state") is not None:
        rng.set_state(parent["sample_rng_state"])

    out.mkdir(parents=True, exist_ok=True)
    parent_valid = heldout(model, valid_ids, windows=eval_windows, seq_len=SEQ_LEN)
    parent_test = heldout(model, test_ids, windows=eval_windows, seq_len=SEQ_LEN)
    parent_core = heldout(model, core_valid_ids, windows=2048, seq_len=SEQ_LEN)
    best_valid = parent_valid
    best_core = parent_core
    best_step = parent_step
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_optimizer = copy.deepcopy(optimizer.state_dict())
    best_torch_rng = torch.get_rng_state().clone()
    best_sample_rng = rng.get_state().clone()
    total_updates = 0
    curve: list[dict] = []
    phase_results: list[dict] = []
    guard_breaches = 0
    stop_reason = "all_stages_completed"
    started = time.monotonic()
    model.train()

    for stage_name, fraction in STAGES:
        pool = progressive_pool(train_ids, fraction)
        phase_start = total_updates
        stage_hash = digest_ids(pool)
        for local_update in range(1, stage_updates + 1):
            x, y = batch_for_lane(lane, pool, core_train_ids, rng, replay_fraction)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_updates += 1

            if local_update % eval_every == 0 or local_update == stage_updates:
                valid_loss = heldout(model, valid_ids, windows=eval_windows, seq_len=SEQ_LEN)
                core_loss = heldout(model, core_valid_ids, windows=2048, seq_len=SEQ_LEN)
                eligible = lane == "language" or core_loss <= parent_core + CORE_REGRESSION_LIMIT
                improved = valid_loss < best_valid - 1e-8
                if eligible and improved:
                    best_valid = valid_loss
                    best_core = core_loss
                    best_step = parent_step + total_updates if lane == "language" else parent_step
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    best_optimizer = copy.deepcopy(optimizer.state_dict())
                    best_torch_rng = torch.get_rng_state().clone()
                    best_sample_rng = rng.get_state().clone()

                if lane != "language" and core_loss > parent_core + CORE_REGRESSION_LIMIT:
                    guard_breaches += 1
                else:
                    guard_breaches = 0

                point = {
                    "stage": stage_name,
                    "stage_fraction": fraction,
                    "stage_updates": local_update,
                    "total_updates": total_updates,
                    "domain_valid_loss": round(valid_loss, 8),
                    "core_language_loss": round(core_loss, 8),
                    "eligible_for_selection": eligible,
                    "selected_valid_loss": round(best_valid, 8),
                    "stage_pool_sha256": stage_hash,
                    "elapsed_seconds": round(time.monotonic() - started, 1),
                }
                curve.append(point)
                print(json.dumps({"event": "eval", "lane": lane, "seed": seed, **point}), flush=True)

                if lane != "language" and guard_breaches >= CORE_REGRESSION_PATIENCE:
                    stop_reason = "language_retention_guard"
                    break
        phase_result = {
            "stage": stage_name,
            "fraction_of_pinned_training_tokens": fraction,
            "stage_updates_completed": total_updates - phase_start,
            "stage_pool_sha256": stage_hash,
        }
        phase_results.append(phase_result)
        print(json.dumps({"event": "stage_complete", "lane": lane, "seed": seed,
                          **phase_result, "stop_reason": stop_reason}), flush=True)
        if stop_reason == "language_retention_guard":
            break

    model.load_state_dict(best_state, strict=True)
    model.eval()
    selected_test = heldout(model, test_ids, windows=eval_windows, seq_len=SEQ_LEN)
    selected_core = heldout(model, core_valid_ids, windows=2048, seq_len=SEQ_LEN)
    optimizer.load_state_dict(best_optimizer)
    torch.set_rng_state(best_torch_rng)
    rng.set_state(best_sample_rng)
    weights_changed = any(not torch.equal(best_state[k], parent["model"][k]) for k in best_state)
    write_checkpoint(out, lane, parent, parent_sha, parent_run, parent_step,
                     model, optimizer, rng, seed, total_updates, best_step, best_valid, manifest)
    task_samples = generate_task_samples(model, tokenizer, lane) if lane in TASK_PROMPTS else []
    if task_samples:
        (out / "task-samples.json").write_text(json.dumps(task_samples, indent=2) + "\n", encoding="utf-8")

    report = {
        "schema": 1,
        "status": "research_only_not_promoted",
        "lane": lane,
        "seed": seed,
        "parent_run": parent_run,
        "parent_checkpoint_sha256": parent_sha,
        "parent_step": parent_step,
        "parent_seed": parent_seed,
        "selected_checkpoint_step": best_step,
        "weights_changed_from_parent": weights_changed,
        "optimizer_state_resumed": resume_optimizer,
        "cumulative_lane_updates": int(parent.get("lane_updates", 0)) + total_updates,
        "candidate_checkpoint_sha256": sha256_file(out / "checkpoint.pt"),
        "updates": total_updates,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "parameters": model.num_parameters,
        "tokenizer_sha256": BASE_TOKENIZER_SHA256,
        "train_sha256": manifest["train_sha256"],
        "valid_sha256": manifest["valid_sha256"],
        "test_sha256": manifest["test_sha256"],
        "progressive_stages": phase_results,
        "eval_every": eval_every,
        "curve": curve,
        "domain_validation_loss_parent": round(parent_valid, 8),
        "domain_validation_loss_candidate": round(best_valid, 8),
        "domain_test_loss_parent": round(parent_test, 8),
        "domain_test_loss_candidate": round(selected_test, 8),
        "core_language_validation_loss_parent": round(parent_core, 8),
        "core_language_validation_loss_candidate": round(selected_core, 8),
        "core_language_delta_candidate_minus_parent": round(selected_core - parent_core, 8),
        "core_language_guard_limit": CORE_REGRESSION_LIMIT if lane != "language" else None,
        "core_language_guard_passed": lane == "language" or selected_core <= parent_core + CORE_REGRESSION_LIMIT,
        "stop_reason": stop_reason,
        "learning_rate": learning_rate,
        "learning_rate_scale": learning_rate_scale,
        "balanced_replay": lane != "language" and abs(replay_fraction - 0.5) < 1e-8,
        "replay_fraction_core": replay_fraction if lane != "language" else None,
        "task_samples_human_reviewed": False,
        "android_integrated": False,
        "promotion": "none; repeatable multi-seed and human task evaluation required",
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "lineage.json").write_text(json.dumps({
        "lane": lane, "seed": seed, "parent_run": parent_run,
        "parent_step": parent_step, "selected_step": best_step,
        "parent_seed": parent_seed, "parent_sha256": parent_sha,
        "candidate_sha256": report["candidate_checkpoint_sha256"],
        "weights_changed_from_parent": weights_changed,
        "optimizer_state_resumed": resume_optimizer,
        "parent_lane_updates": int(parent.get("lane_updates", 0)),
        "cumulative_lane_updates": int(parent.get("lane_updates", 0)) + total_updates,
        "updates_this_run": total_updates, "isolated_lane_weights": True,
        "personal_memory_in_weights": False, "android_integrated": False,
    }, indent=2) + "\n", encoding="utf-8")
    (out / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / "tokenizer.json").write_bytes(tokenizer_path.read_bytes())
    print(json.dumps({"event": "final", **report}, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=("language", "blender", "godot"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage-updates", type=int, default=STAGE_UPDATES)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--parent-sha256")
    parser.add_argument("--parent-run")
    parser.add_argument("--expected-parent-seed", type=int, choices=SEEDS)
    parser.add_argument("--replay-fraction", type=float, default=0.5)
    parser.add_argument("--learning-rate-scale", type=float, default=1.0)
    args = parser.parse_args()
    run(args.lane, args.source, args.out, args.seed, args.stage_updates, args.eval_every,
        args.parent_sha256, args.parent_run, args.expected_parent_seed, args.replay_fraction,
        args.learning_rate_scale)


if __name__ == "__main__":
    main()
