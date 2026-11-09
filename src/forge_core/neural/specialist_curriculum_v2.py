"""Version-2 continuation with expanded CC0 tool lessons and exact new data lineage."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import torch
from tokenizers import Tokenizer

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.seed_v07 import heldout
from forge_core.neural.train import make_batch
from forge_core.neural.multilane_v1 import sha256_file, generate_task_samples

ROOT_PARENT_SHA256 = "a7bcdb996e440f44d7496b887b54d19324927b4296fe9ed2f0f0293c88c63a0a"
TOKENIZER_SHA256 = "4ecd7803d4c369e80bda2e74c30cfb769b7e3a942f931df83383eec9f794f04c"
EXPECTED_PARENTS = {
    "blender": {
        "run": 42445321887, "updates": 1500,
        "sha256": "672db59bfc44d026f26e38d781c84dd24f92882c65b888639ec66a790002ed68",
    },
    "godot": {
        "run": 42445321887, "updates": 1500,
        "sha256": "de7de5052f0a56d821ea3995f1f2e5fbb80d10750e4fb3a564f3c6c4a2a0afdf",
    },
}
ROOT_STEP = 95000
SEQ_LEN = 128
BATCH = 4
EVAL_WINDOWS = 512


def snapshot(parent: dict, model, optimizer, rng, total_updates: int,
             updates_this_run: int, valid_loss: float, manifest: dict,
             source_manifest_sha256: str) -> dict:
    return {
        **parent,
        "lane": parent["lane"],
        "train_sha256": manifest["train_sha256"],
        "valid_sha256": manifest["valid_sha256"],
        "test_sha256": manifest["test_sha256"],
        "lesson_sha256": manifest["lesson_sha256"],
        "source_manifest_sha256": source_manifest_sha256,
        "curriculum_lesson_count": manifest["lesson_count"],
        "root_parent_run": 42143390701,
        "parent_run": 42445321887,
        "parent_step": ROOT_STEP,
        "parent_sha256": ROOT_PARENT_SHA256,
        "immediate_parent_checkpoint_sha256": parent["_immediate_sha"],
        "immediate_parent_lane_updates": parent["_immediate_updates"],
        "step": ROOT_STEP,
        "lane_updates": total_updates,
        "updates_in_current_run": updates_this_run,
        "best_valid_loss": valid_loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "sample_rng_state": rng.get_state(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=("blender", "godot"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2.5e-5)
    args = parser.parse_args()
    if not 1 <= args.steps <= 3000 or not 1 <= args.eval_every <= args.steps:
        parser.error("steps must be 1–3000 and eval-every must be within steps")
    if not 1 <= args.patience <= 6 or not 1e-6 <= args.learning_rate <= 5e-5:
        parser.error("patience or learning rate is outside the controlled range")

    source, out = args.source, args.out
    parent_path = source / "checkpoint.pt"
    parent_sha = sha256_file(parent_path)
    expected = EXPECTED_PARENTS[args.lane]
    if parent_sha != expected["sha256"]:
        raise ValueError(f"Unexpected {args.lane} parent checkpoint SHA-256: {parent_sha}")
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    if parent.get("lane") != args.lane or parent.get("lane_updates") != expected["updates"]:
        raise ValueError("Specialist parent lane/update count mismatch")
    if parent.get("parent_sha256") != ROOT_PARENT_SHA256:
        raise ValueError("Specialist checkpoint does not descend from the verified Forge root")
    if sha256_file(source / "tokenizer.json") != TOKENIZER_SHA256:
        raise ValueError("Forge tokenizer SHA-256 mismatch")
    manifest = json.loads((source / "source-manifest.json").read_text(encoding="utf-8"))
    source_manifest_sha256 = sha256_file(source / "source-manifest.json")
    if manifest.get("lane") != args.lane:
        raise ValueError("Source manifest lane mismatch")
    for split in ("train", "valid", "test"):
        if sha256_file(source / f"{split}.txt") != manifest.get(f"{split}_sha256"):
            raise ValueError(f"{split} data hash does not match the source manifest")
    if manifest.get("lesson_count") != 8 or manifest.get("curriculum_license") != "CC0-1.0, original Forge-Core lesson cards":
        raise ValueError("Unexpected original lesson-card curriculum or license")

    tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    train_ids = torch.tensor(tokenizer.encode((source / "train.txt").read_text(encoding="utf-8")).ids, dtype=torch.long)
    valid_ids = torch.tensor(tokenizer.encode((source / "valid.txt").read_text(encoding="utf-8")).ids, dtype=torch.long)
    test_ids = torch.tensor(tokenizer.encode((source / "test.txt").read_text(encoding="utf-8")).ids, dtype=torch.long)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    model = ByteTransformer(ModelConfig(**parent["configuration"]))
    model.load_state_dict(parent["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(parent["optimizer"]["param_groups"][0]["lr"]))
    optimizer.load_state_dict(parent["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate
    if "torch_rng_state" in parent:
        torch.set_rng_state(parent["torch_rng_state"])
    rng = torch.Generator()
    rng.set_state(parent["sample_rng_state"])

    parent_updates = int(parent["lane_updates"])
    parent_valid = heldout(model, valid_ids, windows=EVAL_WINDOWS, seq_len=SEQ_LEN)
    parent_test = heldout(model, test_ids, windows=EVAL_WINDOWS, seq_len=SEQ_LEN)
    best_valid = parent_valid
    best_updates = 0
    best_model = copy.deepcopy(model.state_dict())
    best_optimizer = copy.deepcopy(optimizer.state_dict())
    best_torch_rng = torch.get_rng_state().clone()
    best_rng = rng.get_state().clone()
    last_model = copy.deepcopy(best_model)
    last_optimizer = copy.deepcopy(best_optimizer)
    last_torch_rng = best_torch_rng.clone()
    last_rng = best_rng.clone()
    curve = [{"updates": 0, "total_lane_updates": parent_updates,
              "valid_token_loss": round(parent_valid, 8)}]
    stale = 0
    updates = 0
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
            score = heldout(model, valid_ids, windows=EVAL_WINDOWS, seq_len=SEQ_LEN)
            curve.append({"updates": updates, "total_lane_updates": parent_updates + updates,
                          "valid_token_loss": round(score, 8),
                          "train_loss": round(float(loss.detach()), 8)})
            last_model = copy.deepcopy(model.state_dict())
            last_optimizer = copy.deepcopy(optimizer.state_dict())
            last_torch_rng = torch.get_rng_state().clone()
            last_rng = rng.get_state().clone()
            if score < best_valid - 1e-4:
                best_valid = score
                best_updates = updates
                best_model = copy.deepcopy(last_model)
                best_optimizer = copy.deepcopy(last_optimizer)
                best_torch_rng = last_torch_rng.clone()
                best_rng = last_rng.clone()
                stale = 0
            else:
                stale += 1
            print(json.dumps({"lane": args.lane, "updates": updates,
                "valid_token_loss": round(score, 8), "best_valid_loss": round(best_valid, 8),
                "stale_evaluations": stale, "patience": args.patience}), flush=True)
            if stale >= args.patience:
                break

    out.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "source-manifest.json", "train.txt", "valid.txt", "test.txt", "ATTRIBUTION.txt"):
        shutil.copy2(source / name, out / name)
    source_parent = dict(parent)
    source_parent["_immediate_sha"] = parent_sha
    source_parent["_immediate_updates"] = parent_updates

    last = snapshot(source_parent, model_from_state(parent, last_model), optimizer_from_state(parent, last_optimizer, args.learning_rate),
                    rng_from_state(last_rng), parent_updates + updates, updates, best_valid,
                    manifest, source_manifest_sha256)
    last["torch_rng_state"] = last_torch_rng
    torch.save(last, out / "last-checkpoint.pt")

    selected = snapshot(source_parent, model_from_state(parent, best_model), optimizer_from_state(parent, best_optimizer, args.learning_rate),
                        rng_from_state(best_rng), parent_updates + best_updates, best_updates, best_valid,
                        manifest, source_manifest_sha256)
    selected["torch_rng_state"] = best_torch_rng
    if best_updates == 0:
        shutil.copy2(parent_path, out / "checkpoint.pt")
        selected_sha = parent_sha
        selected_from = "parent"
    else:
        torch.save(selected, out / "checkpoint.pt")
        selected_sha = sha256_file(out / "checkpoint.pt")
        selected_from = "candidate"

    selected_model = ByteTransformer(ModelConfig(**parent["configuration"]))
    chosen = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=True)
    selected_model.load_state_dict(chosen["model"], strict=True)
    selected_test = heldout(selected_model, test_ids, windows=EVAL_WINDOWS, seq_len=SEQ_LEN)
    selected_valid = heldout(selected_model, valid_ids, windows=EVAL_WINDOWS, seq_len=SEQ_LEN)
    selected_model.eval()
    task_samples = generate_task_samples(selected_model, tokenizer, args.lane)
    (out / "task-samples.json").write_text(json.dumps(task_samples, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema": 1, "status": "research_only_not_promoted", "lane": args.lane,
        "root_parent_run": 42143390701, "root_parent_step": ROOT_STEP,
        "root_parent_sha256": ROOT_PARENT_SHA256,
        "immediate_parent_run": expected["run"], "immediate_parent_checkpoint_sha256": parent_sha,
        "immediate_parent_lane_updates": parent_updates,
        "updates_this_run": updates, "selected_from": selected_from,
        "selected_total_lane_updates": parent_updates + best_updates,
        "selected_checkpoint_sha256": selected_sha,
        "parent_validation_loss": round(parent_valid, 8),
        "selected_validation_loss": round(selected_valid, 8),
        "validation_delta": round(selected_valid - parent_valid, 8),
        "parent_test_loss": round(parent_test, 8),
        "selected_test_loss": round(selected_test, 8),
        "test_delta": round(selected_test - parent_test, 8),
        "train_sha256": manifest["train_sha256"], "valid_sha256": manifest["valid_sha256"],
        "test_sha256": manifest["test_sha256"], "lesson_sha256": manifest["lesson_sha256"],
        "source_manifest_sha256": source_manifest_sha256,
        "parent_training_data_hashes": {k: parent.get(f"{k}_sha256") for k in ("train", "valid", "test")},
        "tokenizer_sha256": TOKENIZER_SHA256,
        "training_config": {"learning_rate_parent": parent["optimizer"]["param_groups"][0]["lr"],
            "learning_rate_used": args.learning_rate, "eval_every": args.eval_every,
            "patience": args.patience, "stop_reason": "early_stopping" if stale >= args.patience else "max_updates"},
        "curve": curve, "task_samples_human_reviewed": False,
        "android_integrated": False, "promoted": False,
        "interpretation": "Fixed manual-page token loss only; not proof of practical Blender or Godot competence.",
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "lineage.json").write_text(json.dumps({
        "lane": args.lane, "root_parent_run": 42143390701, "root_parent_sha256": ROOT_PARENT_SHA256,
        "immediate_parent_run": expected["run"], "immediate_parent_sha256": parent_sha,
        "candidate_sha256": selected_sha, "parent_lane_updates": parent_updates,
        "selected_total_lane_updates": parent_updates + best_updates, "updates_this_run": updates,
        "optimizer_state_resumed": True, "sample_rng_resumed": True,
        "train_sha256": manifest["train_sha256"], "valid_sha256": manifest["valid_sha256"],
        "test_sha256": manifest["test_sha256"], "lesson_sha256": manifest["lesson_sha256"],
        "tokenizer_sha256": TOKENIZER_SHA256, "promoted": False, "android_integrated": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def model_from_state(parent, state):
    model = ByteTransformer(ModelConfig(**parent["configuration"]))
    model.load_state_dict(state, strict=True)
    return model


def optimizer_from_state(parent, state, lr):
    optimizer = torch.optim.AdamW(model_from_state(parent, parent["model"]).parameters(), lr=lr)
    optimizer.load_state_dict(state)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return optimizer


def rng_from_state(state):
    rng = torch.Generator()
    rng.set_state(state)
    return rng


if __name__ == "__main__":
    main()
