"""Measured next-byte pretraining from random weights; no pretrained initialization.

The steps argument counts ADDITIONAL optimizer steps. Each run evaluates a
separate held-out corpus using fixed sampled windows and writes a checkpoint
containing optimizer/RNG states as well as model weights.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.train import make_batch


def fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_corpus(path: Path, seq_len: int) -> tuple[torch.Tensor, str]:
    data = path.read_bytes()
    if len(data) <= seq_len:
        raise ValueError(f"{path} has fewer bytes than a full context and target")
    if len(data) > 16 * 1024 * 1024:
        raise ValueError("CPU experiment is limited to 16 MB per split")
    return torch.tensor(list(data), dtype=torch.long), fingerprint(data)


def configuration(name: str) -> ModelConfig:
    if name == "tiny":
        return ModelConfig(n_layer=1, n_head=4, n_embd=32, max_seq_len=128)
    if name == "base":
        return ModelConfig()
    raise ValueError("Unknown model size")


@torch.no_grad()
def evaluate(model: ByteTransformer, data: torch.Tensor, batch: int,
             seq_len: int, batches: int, seed: int, device: torch.device) -> float:
    """Score the same held-out random windows at every checkpoint."""
    was_training = model.training
    model.eval()
    gen = torch.Generator().manual_seed(seed + 9017)
    losses = []
    for _ in range(batches):
        x, y = make_batch(data, batch, seq_len, gen)
        _, loss = model(x.to(device), y.to(device))
        losses.append(float(loss.detach().cpu()))
    model.train(was_training)
    return sum(losses) / len(losses)


def run(
    *, train_file: Path, valid_file: Path, output: Path,
    model_size: str = "tiny", steps: int = 4, batch_size: int = 2,
    seq_len: int = 32, lr: float = 3e-4, seed: int = 42,
    eval_batches: int = 2, resume: Path | None = None,
    report: Path | None = None, device_name: str = "cpu",
) -> dict:
    if not 1 <= steps <= 100000 or batch_size < 1 or seq_len < 1:
        raise ValueError("Steps, batch size and sequence length must be positive")
    if not (0 < lr < 1) or not 1 <= eval_batches <= 128:
        raise ValueError("Invalid learning rate or evaluation count")
    cfg = configuration(model_size)
    if seq_len > cfg.max_seq_len:
        raise ValueError("Sequence length exceeds model context")
    if device_name not in ("cpu", "cuda") or (device_name == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested device unavailable")
    device = torch.device(device_name)
    train, train_hash = read_corpus(train_file, seq_len)
    valid, valid_hash = read_corpus(valid_file, seq_len)
    if train_hash == valid_hash:
        raise ValueError("Training and validation data are identical")
    if device_name == "cpu":
        torch.set_num_threads(min(2, torch.get_num_threads()))
    torch.manual_seed(seed)
    if device_name == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = ByteTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sample_rng = torch.Generator().manual_seed(seed + 1)
    completed = 0
    history: list[dict] = []
    expected = {
        "schema": 1, "config": asdict(cfg), "seed": seed,
        "train_sha256": train_hash, "valid_sha256": valid_hash,
        "batch_size": batch_size, "seq_len": seq_len, "lr": lr,
    }
    if resume is not None:
        # Only load a known Forge checkpoint from a trusted experiment.
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if any(checkpoint.get(k) != v for k, v in expected.items()):
            raise ValueError("Checkpoint configuration, seed or data hashes mismatch")
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])
        for state in opt.state.values():
            for k, value in state.items():
                if torch.is_tensor(value):
                    state[k] = value.to(device)
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device_name == "cuda":
            if "cuda_rng_state" not in checkpoint:
                raise ValueError("Cannot deterministically resume CUDA without its RNG state")
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        sample_rng.set_state(checkpoint["sample_rng_state"])
        completed = int(checkpoint["step"])
        history = list(checkpoint["history"])
    valid_before = evaluate(model, valid, batch_size, seq_len, eval_batches, seed, device)
    train_loss = math.nan
    model.train()
    for _ in range(steps):
        x, y = make_batch(train, batch_size, seq_len, sample_rng)
        _, loss = model(x.to(device), y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        train_loss = float(loss.detach().cpu())
    completed += steps
    valid_after = evaluate(model, valid, batch_size, seq_len, eval_batches, seed, device)
    row = {
        "global_step": completed, "additional_steps": steps,
        "train_loss_last_batch": round(train_loss, 6),
        "heldout_loss_before": round(valid_before, 6),
        "heldout_loss_after": round(valid_after, 6),
        "heldout_bits_per_byte": round(valid_after / math.log(2), 6),
        "parameter_count": model.num_parameters,
        "train_bytes": train.numel(), "valid_bytes": valid.numel(),
        "train_sha256": train_hash, "valid_sha256": valid_hash,
        "model_size": model_size, "seed": seed,
        "batch_size": batch_size, "seq_len": seq_len,
        "lr": lr, "device": device_name,
        "improved_this_run": valid_after < valid_before,
    }
    history.append(row)
    checkpoint = {
        **expected, "step": completed,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "sample_rng_state": sample_rng.get_state(),
        "history": history,
    }
    if device_name == "cuda":
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    torch.save(checkpoint, tmp)
    os.replace(tmp, output)
    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps({"last_run": row, "history": history}, indent=2) + "\n", encoding="utf-8")
    return row


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--valid", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("artifacts/checkpoint.pt"))
    p.add_argument("--report", type=Path, default=Path("artifacts/metrics.json"))
    p.add_argument("--model-size", choices=("tiny", "base"), default="tiny")
    p.add_argument("--steps", type=int, default=4, help="Additional steps, not total steps")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-batches", type=int, default=2)
    p.add_argument("--resume", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    a = p.parse_args()
    row = run(train_file=a.train, valid_file=a.valid, output=a.out,
              model_size=a.model_size, steps=a.steps, batch_size=a.batch_size,
              seq_len=a.seq_len, lr=a.lr, seed=a.seed, eval_batches=a.eval_batches,
              resume=a.resume, report=a.report, device_name=a.device)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
