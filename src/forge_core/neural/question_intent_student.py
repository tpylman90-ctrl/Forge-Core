"""Controlled question-intent heads over a frozen scratch-trained Forge encoder.

Compares ordinary cross-entropy, a paired supervised-contrastive objective,
and a balanced shuffled-label control. The language model is never updated.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.question_intent_benchmark import LABELS, build, write


class IntentHead(nn.Module):
    def __init__(self, width: int, classes: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(width, 64), nn.Tanh())
        self.classifier = nn.Linear(64, classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.projection(x)
        return self.classifier(z), z


@torch.no_grad()
def encode_questions(model: ByteTransformer, codec: Tokenizer,
                     rows: list[dict[str, str]], batch_size: int = 32) -> torch.Tensor:
    encoded = [codec.encode(row["question"]).ids for row in rows]
    if any(not ids or len(ids) > model.cfg.max_seq_len for ids in encoded):
        raise ValueError("Question tokenization is empty or exceeds model context")
    pad = codec.token_to_id("[UNK]") or 0
    output: list[torch.Tensor] = []
    model.eval()
    for start in range(0, len(rows), batch_size):
        ids = encoded[start:start+batch_size]
        lengths = torch.tensor([len(x) for x in ids], dtype=torch.long)
        width = int(lengths.max())
        tokens = torch.full((len(ids), width), pad, dtype=torch.long)
        for i, sequence in enumerate(ids):
            tokens[i, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        positions = torch.arange(width)
        hidden = model.token(tokens) + model.position(positions)[None, :, :]
        for layer in model.layers:
            hidden = layer(hidden)
        hidden = model.final_norm(hidden)
        output.extend(hidden[i, int(lengths[i])-1].cpu() for i in range(len(ids)))
    return torch.stack(output)


def contrastive_loss(z: torch.Tensor, labels: torch.Tensor,
                     temperature: float = 0.12) -> torch.Tensor:
    z = F.normalize(z, dim=-1)
    similarity = z @ z.T / temperature
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positives = labels[:, None].eq(labels[None, :]) & ~eye
    denominator = similarity.masked_fill(eye, float("-inf"))
    log_probability = similarity - torch.logsumexp(denominator, dim=1, keepdim=True)
    positive_count = positives.sum(dim=1).clamp_min(1)
    return -((log_probability.masked_fill(~positives, 0).sum(dim=1) /
              positive_count).mean())


def accuracy(model: IntentHead, features: torch.Tensor,
             labels: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        logits, _ = model(features)
        pred = logits.argmax(dim=-1)
    confusion = {label: {other: 0 for other in LABELS} for label in LABELS}
    for gold, guess in zip(labels.tolist(), pred.tolist()):
        confusion[LABELS[gold]][LABELS[guess]] += 1
    by_class = {label: confusion[label][label] / max(1, sum(confusion[label].values()))
                for label in LABELS}
    return {
        "correct": int((pred == labels).sum()),
        "total": int(len(labels)),
        "accuracy": float((pred == labels).float().mean()),
        "by_class": by_class,
        "confusion": confusion,
    }


def balanced_batch(labels_by_row: list[int], rng: random.Random,
                   per_class: int = 16) -> list[int]:
    buckets = {i: [] for i in range(len(LABELS))}
    for index, label in enumerate(labels_by_row):
        buckets[label].append(index)
    if any(len(v) < per_class for v in buckets.values()):
        raise ValueError("Training split too small for balanced batches")
    return [index for label in range(len(LABELS))
            for index in rng.sample(buckets[label], per_class)]


def train_arm(train_x: torch.Tensor, train_y: list[int],
              dev_x: torch.Tensor, dev_y: torch.Tensor, *,
              seed: int, arm: str, updates: int = 600) -> tuple[IntentHead, dict]:
    torch.manual_seed(seed)
    randomizer = random.Random(seed)
    observed_y = list(train_y)
    if arm == "shuffled":
        shuffled = list(observed_y)
        randomizer.shuffle(shuffled)
        observed_y = shuffled
    head = IntentHead(train_x.shape[-1], len(LABELS))
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    target = torch.tensor(observed_y, dtype=torch.long)
    best_score = -1.0
    best_state = None
    best_step = 0
    for step in range(1, updates + 1):
        indices = balanced_batch(observed_y, randomizer)
        x = train_x[indices]
        y = target[indices]
        logits, z = head(x)
        ce = F.cross_entropy(logits, y)
        loss = ce if arm in ("plain", "shuffled") else (
            0.5 * ce + 0.5 * contrastive_loss(z, y)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 10 == 0:
            score = accuracy(head, dev_x, dev_y)["accuracy"]
            if score > best_score:
                best_score = score
                best_step = step
                best_state = {key: value.detach().clone()
                              for key, value in head.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No development checkpoint selected")
    head.load_state_dict(best_state)
    return head, {"best_dev_accuracy": best_score, "best_step": best_step}


def label_ids(rows: list[dict[str, str]]) -> torch.Tensor:
    return torch.tensor([LABELS.index(row["label"]) for row in rows], dtype=torch.long)


def run(checkpoint_dir: Path, out: Path, *, updates: int = 600) -> dict:
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    tokenizer_path = checkpoint_dir / "tokenizer.json"
    before_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if ckpt.get("schema") != 3 or ckpt.get("step", 0) < 1:
        raise ValueError("Expected an audited Forge v0.7 research checkpoint")
    source_manifest = json.loads((checkpoint_dir / "source-manifest.json").read_text())
    if source_manifest.get("license") != "cdla-sharing-1.0":
        raise ValueError("Checkpoint source license is not the reviewed public license")
    for key in ("revision", "train_sha256", "valid_sha256"):
        if ckpt.get(key) != source_manifest.get(key):
            raise ValueError(f"Checkpoint/source manifest mismatch: {key}")
    cfg = ModelConfig(**ckpt["configuration"])
    encoder = ByteTransformer(cfg)
    encoder.load_state_dict(ckpt["model"])
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    codec = Tokenizer.from_file(str(tokenizer_path))

    splits = build()
    out.mkdir(parents=True, exist_ok=True)
    benchmark_dir = out / "benchmark"
    manifest = write(benchmark_dir)
    train_rows, dev_rows, final_rows = (
        splits["train"], splits["development"], splits["final"]
    )
    train_x = encode_questions(encoder, codec, train_rows)
    dev_x = encode_questions(encoder, codec, dev_rows)
    final_x = encode_questions(encoder, codec, final_rows)
    train_y = [LABELS.index(row["label"]) for row in train_rows]
    dev_y = label_ids(dev_rows)
    final_y = label_ids(final_rows)

    results = []
    for seed in (1201, 1202, 1203):
        for arm in ("plain", "contrastive", "shuffled"):
            head, selection = train_arm(
                train_x, train_y, dev_x, dev_y, seed=seed,
                arm=arm, updates=updates,
            )
            scored = accuracy(head, final_x, final_y)
            results.append({
                "seed": seed, "arm": arm, **selection,
                "final": scored,
            })
            torch.save(head.state_dict(), out / f"head-{arm}-{seed}.pt")
    after_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if after_hash != before_hash:
        raise ValueError("Frozen Forge language checkpoint changed during study")
    report = {
        "schema": 1,
        "benchmark": manifest,
        "source_checkpoint_step": ckpt["step"],
        "source_checkpoint_sha256": before_hash,
        "encoder_frozen": True,
        "updates_per_arm_seed": updates,
        "arms": ["plain_cross_entropy", "paired_contrastive_plus_ce",
                 "balanced_shuffled_label_control"],
        "chance_baseline": 1 / len(LABELS),
        "results": results,
        "promotion": "NONE; one internal benchmark is diagnostic only",
        "limitations": [
            "Question-only synthetic wording does not test story reading or event-role binding.",
            "Final split is public source and must not be reused as an untouched future exam.",
            "The random head is research-only; frozen language-model weights remain unchanged.",
        ],
    }
    (out / "question-intent-results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=600)
    args = parser.parse_args()
    if not 1 <= args.updates <= 2000:
        raise ValueError("Updates outside bounded range")
    print(json.dumps(run(args.checkpoint_dir, args.out, updates=args.updates), indent=2))


if __name__ == "__main__":
    main()
