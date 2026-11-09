"""Conservative follow-on continuation from the selected 107k language candidate.

Resumes the validation-best 107k branch and its optimizer moments.
Selects the best checkpoint by validation and uses a fresh 20 MiB-offset
TinyStories range for final reporting.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch
from tokenizers import Tokenizer

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.multilane_v1 import load_language_data
from forge_core.neural.seed_v07 import heldout
from forge_core.neural.train import make_batch

PARENT_RUN = 42539881351
PARENT_STEP = 107000
PARENT_SHA256 = "ed2f1395b52c3ae2c674e00f3b86f82ec26685e2d1c631f6eb975ddb3f05fe0f"
PARENT_LR = 1.25e-5
TEST_OFFSET_BYTES = 20 * 1024 * 1024
TOKENIZER_SHA256 = "4ecd7803d4c369e80bda2e74c30cfb769b7e3a942f931df83383eec9f794f04c"
SEQ_LEN = 128
BATCH = 4
WINDOWS = 2048


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_state(path: Path, parent: dict, model: ByteTransformer, optimizer,
               rng: torch.Generator, updates: int, lr: float, curve: list[dict],
               sampler_seed: int) -> None:
    checkpoint = {
        **parent,
        "step": PARENT_STEP + updates,
        "lr": lr,
        "parent_run": PARENT_RUN,
        "parent_step": PARENT_STEP,
        "parent_sha256": PARENT_SHA256,
        "sampler_seed": sampler_seed,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "sample_rng_state": rng.get_state(),
        "curve": curve,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp)
    os.replace(tmp, path)


def evaluate(model: ByteTransformer, tokens: torch.Tensor) -> float:
    return heldout(model, tokens, windows=WINDOWS, seq_len=SEQ_LEN, batch=BATCH)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.25e-5)
    parser.add_argument("--sampler-seed", type=int, choices=(0,), default=0,
                        help="0 resumes the selected parent's saved sample RNG")
    args = parser.parse_args()
    if not 1 <= args.steps <= 8000 or not 1 <= args.eval_every <= args.steps:
        parser.error("steps must be 1–8000 and eval-every must be within steps")
    if not 1 <= args.patience <= 4 or not 1e-6 <= args.lr < 3e-4:
        parser.error("patience or learning rate outside the conservative approved range")

    source, out = args.source, args.out
    parent_path = source / "checkpoint.pt"
    parent_sha = digest(parent_path)
    if parent_sha != PARENT_SHA256:
        raise ValueError(f"Unexpected parent checkpoint SHA-256: {parent_sha}")
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    if parent.get("step") != PARENT_STEP or parent.get("seed") != 43:
        raise ValueError("Expected Forge Seed v0.7's verified 106k language branch")
    if parent.get("lr") != PARENT_LR or parent.get("batch") != BATCH or parent.get("seq_len") != SEQ_LEN:
        raise ValueError("Parent optimizer/training configuration differs from the verified run")
    if digest(source / "tokenizer.json") != TOKENIZER_SHA256:
        raise ValueError("Forge tokenizer hash does not match the verified lineage")

    tokenizer_hash, train_ids, valid_ids, test_ids, manifest = load_language_data(
        source, test_offset_bytes=TEST_OFFSET_BYTES
    )
    if tokenizer_hash != TOKENIZER_SHA256:
        raise ValueError("Loaded tokenizer changed during data preparation")
    if manifest["revision"] != "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64":
        raise ValueError("TinyStories revision is not the approved immutable revision")
    if manifest["license"] != "cdla-sharing-1.0":
        raise ValueError("TinyStories license metadata changed")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "tokenizer.json", out / "tokenizer.json")
    shutil.copy2(source / "checkpoint.pt", out / "parent-checkpoint.pt")
    for name in ("train.txt", "valid.txt"):
        shutil.copy2(source / name, out / name)
    shutil.copy2(source / "test.txt", out / "test.txt")
    (out / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "ATTRIBUTION.txt").write_text(
        "TinyStories by Ronen Eldan; license: CDLA-Sharing-1.0.\n"
        "Dataset revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64.\n"
        "https://huggingface.co/datasets/roneneldan/TinyStories\n",
        encoding="utf-8",
    )

    torch.set_num_threads(min(2, torch.get_num_threads()))
    model = ByteTransformer(ModelConfig(**parent["configuration"]))
    model.load_state_dict(parent["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(parent["lr"]))
    optimizer.load_state_dict(parent["optimizer"])
    # Retain AdamW moments exactly; deliberately reduce only the step size.
    for group in optimizer.param_groups:
        group["lr"] = args.lr
    torch.set_rng_state(parent["torch_rng_state"])
    rng = torch.Generator()
    if args.sampler_seed == 0:
        rng.set_state(parent["sample_rng_state"])
        effective_sampler_seed = int(parent.get("sampler_seed", parent["seed"]))
    else:
        rng.manual_seed(args.sampler_seed + 1)
        effective_sampler_seed = args.sampler_seed

    parent_valid = evaluate(model, valid_ids)
    parent_test = evaluate(model, test_ids)
    best_valid = parent_valid
    best_updates = 0
    best_model = copy.deepcopy(model.state_dict())
    best_optimizer = copy.deepcopy(optimizer.state_dict())
    best_torch_rng = torch.get_rng_state().clone()
    best_sample_rng = rng.get_state().clone()
    curve = [{"updates": 0, "global_step": PARENT_STEP,
              "valid_loss_2048": round(parent_valid, 8)}]
    stale_evals = 0
    updates = 0
    last_model = copy.deepcopy(best_model)
    last_optimizer = copy.deepcopy(best_optimizer)
    last_torch_rng = best_torch_rng.clone()
    last_sample_rng = best_sample_rng.clone()

    model.train()
    while updates < args.steps:
        x, y = make_batch(train_ids, BATCH, SEQ_LEN, rng)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        updates += 1
        if updates % args.eval_every == 0 or updates == args.steps:
            score = evaluate(model, valid_ids)
            point = {"updates": updates, "global_step": PARENT_STEP + updates,
                     "valid_loss_2048": round(score, 8),
                     "train_loss": round(float(loss.detach()), 8)}
            curve.append(point)
            last_model = copy.deepcopy(model.state_dict())
            last_optimizer = copy.deepcopy(optimizer.state_dict())
            last_torch_rng = torch.get_rng_state().clone()
            last_sample_rng = rng.get_state().clone()
            save_state(out / "last-checkpoint.pt", parent, model, optimizer, rng,
                       updates, args.lr, curve, effective_sampler_seed)
            improved = score < best_valid - 1e-4
            if improved:
                best_valid = score
                best_updates = updates
                best_model = copy.deepcopy(last_model)
                best_optimizer = copy.deepcopy(last_optimizer)
                best_torch_rng = last_torch_rng.clone()
                best_sample_rng = last_sample_rng.clone()
                save_state(out / "best-checkpoint.pt", parent, model, optimizer,
                           rng, updates, args.lr, curve, effective_sampler_seed)
                stale_evals = 0
            else:
                stale_evals += 1
            print(json.dumps({
                "progress": point, "best_valid_loss_2048": round(best_valid, 8),
                "best_global_step": PARENT_STEP + best_updates,
                "stale_evals": stale_evals, "patience": args.patience,
                "early_stop": stale_evals >= args.patience,
            }), flush=True)
            if stale_evals >= args.patience:
                break

    # Keep the last actual update separately. The selected checkpoint is the
    # lowest-validation state; if no update beats the parent, select the parent.
    torch.save({
        **parent, "step": PARENT_STEP + updates, "lr": args.lr,
        "parent_run": PARENT_RUN, "parent_step": PARENT_STEP,
        "parent_sha256": PARENT_SHA256,
        "sampler_seed": effective_sampler_seed,
        "model": last_model, "optimizer": last_optimizer,
        "torch_rng_state": last_torch_rng, "sample_rng_state": last_sample_rng,
        "curve": curve,
    }, out / "last-checkpoint.pt")
    selected_from = "candidate" if best_updates else "parent"
    if best_updates:
        selected = {
            **parent, "step": PARENT_STEP + best_updates, "lr": args.lr,
            "parent_run": PARENT_RUN, "parent_step": PARENT_STEP,
            "parent_sha256": PARENT_SHA256,
            "sampler_seed": effective_sampler_seed,
            "model": best_model, "optimizer": best_optimizer,
            "torch_rng_state": best_torch_rng, "sample_rng_state": best_sample_rng,
            "curve": curve,
        }
        torch.save(selected, out / "checkpoint.pt")
        torch.save(selected, out / "best-checkpoint.pt")
    else:
        selected = parent
        shutil.copy2(parent_path, out / "checkpoint.pt")
        shutil.copy2(parent_path, out / "best-checkpoint.pt")

    selected_model = ByteTransformer(ModelConfig(**selected["configuration"]))
    selected_model.load_state_dict(selected["model"], strict=True)
    candidate_test = evaluate(selected_model, test_ids)
    last_model_obj = ByteTransformer(ModelConfig(**parent["configuration"]))
    last_model_obj.load_state_dict(last_model, strict=True)
    last_test = evaluate(last_model_obj, test_ids)

    result = {
        "schema": 1, "status": "research_candidate_not_promoted",
        "parent_run": PARENT_RUN, "parent_step": PARENT_STEP,
        "parent_checkpoint_sha256": parent_sha,
        "parent_validation_loss_2048": round(parent_valid, 8),
        "parent_independent_test_loss_2048": round(parent_test, 8),
        "actual_updates_completed": updates, "last_global_step": PARENT_STEP + updates,
        "selected_global_step": PARENT_STEP + best_updates,
        "selected_from": selected_from,
        "selected_checkpoint_sha256": digest(out / "checkpoint.pt"),
        "last_checkpoint_sha256": digest(out / "last-checkpoint.pt"),
        "selected_validation_loss_2048": round(best_valid, 8),
        "selected_validation_delta": round(best_valid - parent_valid, 8),
        "selected_independent_test_loss_2048": round(candidate_test, 8),
        "selected_independent_test_delta": round(candidate_test - parent_test, 8),
        "last_independent_test_loss_2048": round(last_test, 8),
        "last_independent_test_delta": round(last_test - parent_test, 8),
        "learning_rate_parent": parent["lr"], "learning_rate_used": args.lr,
        "sampler_seed": effective_sampler_seed,
        "requested_sampler_seed": args.sampler_seed,
        "sampler_rng_policy": ("resumed_parent_rng" if args.sampler_seed == 0
                               else f"manual_seed({args.sampler_seed + 1})"),
        "optimizer_moments_resumed": True, "optimizer_moments_reset": False,
        "tokenizer_sha256": tokenizer_hash,
        "train_sha256": manifest["train_sha256"],
        "valid_sha256": manifest["valid_sha256"],
        "independent_test_sha256": manifest["test_sha256"],
        "validation_and_test_windows": WINDOWS, "context_tokens": SEQ_LEN,
        "early_stop_patience": args.patience, "curve": curve,
        "android_integrated": False,
        "interpretation": "Fixed-split next-token diagnostics only; not evidence of comprehension or reasoning.",
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "lineage.json").write_text(json.dumps({
        "kind": "continued_core_language_checkpoint",
        "status": "research_candidate_not_promoted",
        "parent_run": PARENT_RUN, "parent_step": PARENT_STEP,
        "parent_sha256": parent_sha,
        "selected_step": PARENT_STEP + best_updates,
        "last_step": PARENT_STEP + updates,
        "actual_updates": updates, "sampler_seed": effective_sampler_seed,
        "requested_sampler_seed": args.sampler_seed,
        "sampler_rng_policy": ("resumed_parent_rng" if args.sampler_seed == 0
                               else f"manual_seed({args.sampler_seed + 1})"),
        "learning_rate_parent": parent["lr"],
        "learning_rate_used": args.lr,
        "optimizer_moments_resumed": True, "source_data_reused_exactly": True,
        "tokenizer_sha256": tokenizer_hash, "selected_checkpoint_sha256": digest(out / "checkpoint.pt"),
        "last_checkpoint_sha256": digest(out / "last-checkpoint.pt"),
        "android_integrated": False,
    }, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
