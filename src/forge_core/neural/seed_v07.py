"""Forge Seed v0.7: entirely scratch-trained ByteLevel-BPE and Transformer.

Training is public-data only, bounded, reproducible and evaluated on the
upstream validation file, never on observation logs or private memories.
Tokenizer and neural weights are both initialized/trained FROM SCRATCH.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.train import make_batch

PROMPTS = (
    "Once upon a time, there was a little",
    "The small dog went to the",
    "A girl found a red ball and",
    "One day, a boy saw a bird",
    "The kind teacher asked the children",
    "The rabbit was very happy because",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_manifest(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != 1 or manifest.get("revision") != "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64":
        raise ValueError("Unpinned or unsupported source manifest")
    if manifest.get("license") != "cdla-sharing-1.0":
        raise ValueError("Source license not reviewed")
    for split in ("train", "valid"):
        if sha256(directory / f"{split}.txt") != manifest.get(f"{split}_sha256"):
            raise ValueError(f"{split} data differs from manifest")
        if manifest.get(f"{split}_document_count", 0) < 30:
            raise ValueError("Insufficient independent complete stories")
    return manifest


def learn_tokenizer(train_path: Path, output_path: Path, vocab: int = 2048) -> Tokenizer:
    if not 512 <= vocab <= 8192:
        raise ValueError("Vocabulary outside approved experiment range")
    codec = Tokenizer(models.BPE(unk_token="[UNK]"))
    codec.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    codec.decoder = decoders.ByteLevel()
    codec.train([str(train_path)], trainers.BpeTrainer(
        vocab_size=vocab, min_frequency=2, special_tokens=["[UNK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    ))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    codec.save(str(output_path))
    return codec


@torch.no_grad()
def heldout(model: ByteTransformer, tokens: torch.Tensor, *, windows: int,
            seq_len: int, batch: int = 4) -> float:
    if tokens.numel() <= seq_len + 1:
        raise ValueError("Not enough held-out data")
    was_train = model.training
    model.eval()
    starts = torch.linspace(0, tokens.numel() - seq_len - 1, steps=windows).long()
    total, count = 0.0, 0
    for start_idx in range(0, windows, batch):
        indices = starts[start_idx:start_idx+batch].tolist()
        x = torch.stack([tokens[i:i+seq_len] for i in indices])
        y = torch.stack([tokens[i+1:i+seq_len+1] for i in indices])
        _, loss = model(x, y)
        total += float(loss) * len(indices)
        count += len(indices)
    model.train(was_train)
    return total / count


@torch.no_grad()
def generate_samples(model: ByteTransformer, codec: Tokenizer, seed: int,
                     new_tokens: int = 80) -> list[dict]:
    result = []
    was_train = model.training
    model.eval()
    with torch.random.fork_rng():
        for i, prefix in enumerate(PROMPTS):
            torch.manual_seed(seed + 6000 + i)
            prompt_ids = codec.encode(prefix).ids
            generated = model.generate(torch.tensor([prompt_ids], dtype=torch.long),
                                       new_tokens=new_tokens, temperature=0.75)
            continuation = codec.decode(generated[0].tolist()[len(prompt_ids):],
                                        skip_special_tokens=True)
            result.append({"prompt": prefix, "continuation": continuation})
    model.train(was_train)
    return result


def train(*, source: Path, out: Path, steps: int = 12000, seed: int = 42,
          lr: float = 3e-4, batch: int = 4, seq_len: int = 128,
          vocab: int = 2048, model_size: str = "seed",
          eval_every: int = 2000, resume: Path | None = None) -> dict:
    if not 1 <= steps <= 40000 or not 1 <= batch <= 16:
        raise ValueError("Training steps/batch outside bounded range")
    if not 32 <= seq_len <= 256 or not 1 <= eval_every <= steps:
        raise ValueError("Invalid context or evaluation interval")
    if not 0 < lr < 0.01:
        raise ValueError("Learning rate outside bounded range")
    if model_size not in ("tiny", "seed"):
        raise ValueError("Only controlled tiny and seed training supported")
    manifest = verified_manifest(source)
    out.mkdir(parents=True, exist_ok=True)
    codec_path = out / "tokenizer.json"
    codec = learn_tokenizer(source / "train.txt", codec_path, vocab)
    train_ids = torch.tensor(codec.encode((source / "train.txt").read_text()).ids,
                             dtype=torch.long)
    valid_ids = torch.tensor(codec.encode((source / "valid.txt").read_text()).ids,
                             dtype=torch.long)
    if train_ids.numel() <= seq_len * batch or valid_ids.numel() <= seq_len * batch:
        raise ValueError("Dataset cannot support the requested context")
    cfg = ModelConfig(vocab_size=codec.get_vocab_size(), max_seq_len=256,
                      n_layer=4 if model_size=="seed" else 1,
                      n_head=4, n_embd=192 if model_size=="seed" else 32)
    torch.set_num_threads(min(2, torch.get_num_threads()))
    torch.manual_seed(seed)
    random.seed(seed)
    model = ByteTransformer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed + 1)
    expected = {
        "schema": 3, "configuration": asdict(cfg),
        "train_sha256": manifest["train_sha256"],
        "valid_sha256": manifest["valid_sha256"],
        "tokenizer_sha256": sha256(codec_path),
        "revision": manifest["revision"],
        "seed": seed, "lr": lr, "batch": batch, "seq_len": seq_len,
    }
    completed = 0
    curve = []
    if resume:
        ckpt = torch.load(resume, map_location="cpu", weights_only=True)
        for key, value in expected.items():
            if ckpt.get(key) != value:
                raise ValueError(f"Incompatible resume checkpoint field {key}")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        torch.set_rng_state(ckpt["torch_rng_state"])
        generator.set_state(ckpt["sample_rng_state"])
        completed = int(ckpt["step"])
        curve = list(ckpt["curve"])
    start = time.monotonic()
    initial_loss = heldout(model, valid_ids, windows=64, seq_len=seq_len)
    curve.append({"step": completed, "valid_token_loss": round(initial_loss, 6)})
    model.train()
    for step in range(1, steps + 1):
        x, y = make_batch(train_ids, batch, seq_len, generator)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % eval_every == 0 or step == steps:
            score = heldout(model, valid_ids, windows=64, seq_len=seq_len)
            point = {"step": completed + step, "valid_token_loss": round(score, 6),
                     "train_loss": round(float(loss.detach()), 6)}
            curve.append(point)
            print(json.dumps({"progress": point}), flush=True)
    end_step = completed + steps
    final_loss = heldout(model, valid_ids, windows=256, seq_len=seq_len)
    # Sampled token-NLL multiplied by token-density. This is an ESTIMATE,
    # not exact whole-validation bits per byte.
    estimated_bpb = final_loss * valid_ids.numel() / (
        (source / "valid.txt").stat().st_size * math.log(2))
    samples = generate_samples(model, codec, seed)
    report = {
        "status": "research_not_preschool_certified", "source": manifest,
        "parameters": model.num_parameters, "tokenizer_vocab": codec.get_vocab_size(),
        "tokenizer_sha256": sha256(codec_path), "train_token_count": train_ids.numel(),
        "valid_token_count": valid_ids.numel(), "seed": seed, "batch": batch,
        "context_tokens": seq_len, "steps_total": end_step,
        "additional_steps": steps, "learning_rate": lr,
        "validation_token_loss_start": round(initial_loss, 6),
        "validation_token_loss_after": round(final_loss, 6),
        "estimated_bits_per_byte": round(estimated_bpb, 6),
        "estimated_bpb_caveat": "Derived from 256 sampled token windows; not exact full-corpus BPE/byte comparison.",
        "curve": curve, "generation_samples": samples,
        "wall_seconds": round(time.monotonic() - start, 2),
        "note": "Language abilities must be judged on unseen sentences and instructions separately.",
    }
    ckpt = {
        **expected, "step": end_step, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "sample_rng_state": generator.get_state(), "curve": curve,
    }
    tmp = out / "checkpoint.pt.tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, out / "checkpoint.pt")
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("source","generation_samples","curve")},
                     indent=2), flush=True)
    for sample in samples:
        print("SAMPLE", json.dumps(sample, ensure_ascii=False), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("artifacts/preschool"))
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-size", choices=("tiny","seed"), default="seed")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--vocab", type=int, default=2048)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--resume", type=Path)
    a = p.parse_args()
    train(source=a.source, out=a.out, steps=a.steps, seed=a.seed,
          model_size=a.model_size, lr=a.lr, batch=a.batch, seq_len=a.seq_len,
          vocab=a.vocab, eval_every=a.eval_every, resume=a.resume)


if __name__ == "__main__":
    main()
