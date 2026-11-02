"""Small reproducible next-byte training experiment.

Provide an approved local UTF-8 corpus. No automatic downloads or personal data.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import random

import torch

from forge_core.neural.model import ByteTransformer, ModelConfig


def make_batch(data: torch.Tensor, batch_size: int, seq_len: int, generator: torch.Generator):
    if data.numel() <= seq_len:
        raise ValueError("Training file must contain more bytes than seq_len")
    indices = torch.randint(0, data.numel() - seq_len, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + seq_len] for i in indices])
    y = torch.stack([data[i + 1 : i + 1 + seq_len] for i in indices])
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=Path, required=True, help="Approved local UTF-8 corpus")
    parser.add_argument("--out", type=Path, default=Path("artifacts/forge-v0.1.pt"))
    parser.add_argument("--model-size", choices=("tiny", "base"), default="base")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1 or args.seq_len < 1 or args.lr <= 0:
        parser.error("steps, batch-size, seq-len and lr must be positive")
    cfg = (
        ModelConfig(n_layer=1, n_head=4, n_embd=32, max_seq_len=128)
        if args.model_size == "tiny" else ModelConfig()
    )
    if args.seq_len > cfg.max_seq_len:
        parser.error(f"seq-len exceeds model context of {cfg.max_seq_len}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    # This path must be local and approved. Raw text never enters the checkpoint.
    corpus = args.text.read_text(encoding="utf-8").encode("utf-8")
    data = torch.tensor(list(corpus), dtype=torch.long)
    if len(data) <= args.seq_len:
        parser.error("Corpus needs more UTF-8 bytes than seq-len")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    model = ByteTransformer(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    print(f"Parameters: {model.num_parameters:,}; dataset bytes: {len(corpus):,}")
    loss_value = float("nan")
    for step in range(1, args.steps + 1):
        x, y = make_batch(data, args.batch_size, args.seq_len, g)
        x, y = x.to(args.device), y.to(args.device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        loss_value = float(loss.detach().cpu())
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(f"step={step} train_loss={loss_value:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"config": asdict(cfg), "step": args.steps, "seed": args.seed,
         "model": model.cpu().state_dict(), "train_loss": loss_value},
        args.out,
    )
    print(f"Saved weights to {args.out} (no raw corpus)")


if __name__ == "__main__":
    main()
