"""Adaptive continuation for Forge Seed v0.7.

Frequent validation checkpoints and rollback/LR reduction on sustained regression.
The resulting checkpoint remains a research candidate and never promotes itself.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import time

import torch
from tokenizers import Tokenizer

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.seed_v07 import (
    generate_samples, heldout, sha256, verified_manifest,
)
from forge_core.neural.train import make_batch


def atomic_save(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def train(*, source: Path, out: Path, steps: int, seed: int = 43,
          lr: float = 3e-4, batch: int = 4, seq_len: int = 128,
          eval_every: int = 250, regression_patience: int = 2,
          regression_threshold: float = 0.01, lr_floor: float = 1e-5) -> dict:
    if not 1 <= steps <= 10000 or not 1 <= eval_every <= steps:
        raise ValueError("Training budget/evaluation interval outside bounds")
    if regression_patience < 1 or not 0 < regression_threshold <= 0.10:
        raise ValueError("Invalid regression trigger")
    if not 0 < lr_floor <= lr or not 0 < lr < 0.01:
        raise ValueError("Invalid learning-rate range")

    manifest = verified_manifest(source)
    out.mkdir(parents=True, exist_ok=True)
    codec_path = source / "tokenizer.json"
    codec = Tokenizer.from_file(str(codec_path))
    tokenizer_hash = sha256(codec_path)
    train_ids = torch.tensor(codec.encode((source / "train.txt").read_text()).ids,
                             dtype=torch.long)
    valid_ids = torch.tensor(codec.encode((source / "valid.txt").read_text()).ids,
                             dtype=torch.long)
    parent = torch.load(source / "checkpoint.pt", map_location="cpu", weights_only=True)
    if parent.get("tokenizer_sha256") != tokenizer_hash:
        raise ValueError("Parent tokenizer hash mismatch")
    if parent.get("train_sha256") != manifest["train_sha256"] or parent.get("valid_sha256") != manifest["valid_sha256"]:
        raise ValueError("Parent data hash mismatch")
    if parent.get("seed") != seed or parent.get("batch") != batch or parent.get("seq_len") != seq_len:
        raise ValueError("Parent training configuration mismatch")

    torch.set_num_threads(min(2, torch.get_num_threads()))
    torch.manual_seed(seed)
    random.seed(seed)
    cfg = ModelConfig(**parent["configuration"])
    model = ByteTransformer(cfg)
    model.load_state_dict(parent["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer.load_state_dict(parent["optimizer"])
    generator = torch.Generator().manual_seed(seed + 1)
    torch.set_rng_state(parent["torch_rng_state"])
    generator.set_state(parent["sample_rng_state"])
    start_step = int(parent["step"])
    current_step = start_step
    attempted = 0
    curve: list[dict] = []
    events: list[dict] = []

    expected = {
        "schema": 3, "configuration": asdict(cfg),
        "train_sha256": manifest["train_sha256"],
        "valid_sha256": manifest["valid_sha256"],
        "tokenizer_sha256": tokenizer_hash,
        "revision": manifest["revision"], "seed": seed,
        "lr": lr, "batch": batch, "seq_len": seq_len,
    }

    def snapshot(step: int, attempts: int) -> dict:
        return {
            **expected, "step": step, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "sample_rng_state": generator.get_state(),
            "curve": list(curve), "recovery_events": list(events),
            "training_updates_attempted": attempts,
        }

    initial = heldout(model, valid_ids, windows=64, seq_len=seq_len)
    best_loss = initial
    curve.append({"step": current_step, "attempt": 0,
                  "valid_token_loss": round(initial, 8), "kind": "parent"})
    atomic_save(snapshot(current_step, attempted), out / "best.pt")
    best_step = current_step
    regressions = 0
    started = time.monotonic()
    model.train()

    while attempted < steps:
        x, y = make_batch(train_ids, batch, seq_len, generator)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        attempted += 1
        current_step += 1

        if attempted % eval_every == 0 or attempted == steps:
            score = heldout(model, valid_ids, windows=64, seq_len=seq_len)
            point = {"step": current_step, "attempt": attempted,
                     "valid_token_loss": round(score, 8),
                     "train_loss": round(float(loss.detach()), 8)}
            curve.append(point)
            print(json.dumps({"progress": point}), flush=True)

            if score < best_loss:
                best_loss, best_step, regressions = score, current_step, 0
                atomic_save(snapshot(current_step, attempted), out / "best.pt")
            elif score > best_loss * (1.0 + regression_threshold):
                regressions += 1
            else:
                regressions = 0

            # Always retain a recovery point, even when it is not the best.
            atomic_save(snapshot(current_step, attempted), out / "latest.pt")
            if regressions >= regression_patience:
                best = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
                model.load_state_dict(best["model"])
                optimizer.load_state_dict(best["optimizer"])
                torch.set_rng_state(best["torch_rng_state"])
                generator.set_state(best["sample_rng_state"])
                old_lr = min(float(group["lr"]) for group in optimizer.param_groups)
                new_lr = max(lr_floor, old_lr / 2.0)
                if new_lr >= old_lr:
                    events.append({"attempt": attempted, "action": "stop_at_lr_floor",
                                   "best_step": best_step, "best_loss": best_loss})
                    print(json.dumps({"recovery": events[-1]}), flush=True)
                    break
                for group in optimizer.param_groups:
                    group["lr"] = new_lr
                current_step = best_step
                events.append({"attempt": attempted, "action": "rollback_and_lower_lr",
                               "from_step": point["step"], "to_step": best_step,
                               "old_lr": old_lr, "new_lr": new_lr,
                               "best_loss": round(best_loss, 8)})
                regressions = 0
                print(json.dumps({"recovery": events[-1]}), flush=True)

    # The candidate delivered downstream is always the best held-out checkpoint.
    best = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])
    optimizer.load_state_dict(best["optimizer"])
    current_step = int(best["step"])
    final_loss = heldout(model, valid_ids, windows=256, seq_len=seq_len)
    estimated_bpb = final_loss * valid_ids.numel() / (
        (source / "valid.txt").stat().st_size * __import__("math").log(2))
    (out / "tokenizer.json").write_bytes(codec_path.read_bytes())
    samples = generate_samples(model, codec, seed)
    report = {
        "status": "research_candidate_not_promoted",
        "parent_step": start_step, "steps_total": current_step,
        "training_updates_attempted": attempted,
        "parameters": model.num_parameters, "seed": seed,
        "initial_learning_rate": lr, "regression_lr_floor": lr_floor,
        "checkpoint_interval_updates": eval_every,
        "regression_threshold": regression_threshold,
        "regression_patience": regression_patience,
        "recovery_events": events, "curve": curve,
        "validation_token_loss_start": round(initial, 8),
        "validation_token_loss_after": round(final_loss, 8),
        "estimated_bits_per_byte": round(estimated_bpb, 6),
        "estimated_bpb_caveat": "256 sampled token windows; not exact whole-corpus bits per byte.",
        "generation_samples": samples,
        "wall_seconds": round(time.monotonic() - started, 2),
        "interpretation": "Held-out text prediction only; not proof of comprehension or reasoning.",
    }
    atomic_save(best, out / "checkpoint.pt")
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("curve", "generation_samples")}, indent=2), flush=True)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--regression-patience", type=int, default=2)
    p.add_argument("--regression-threshold", type=float, default=0.01)
    p.add_argument("--lr-floor", type=float, default=1e-5)
    a = p.parse_args()
    train(source=a.source, out=a.out, steps=a.steps, seed=a.seed, lr=a.lr,
          batch=a.batch, seq_len=a.seq_len, eval_every=a.eval_every,
          regression_patience=a.regression_patience,
          regression_threshold=a.regression_threshold, lr_floor=a.lr_floor)


if __name__ == "__main__":
    main()
