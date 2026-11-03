"""Audit Forge Seed v0.7 before retaining a public-data research checkpoint.

Do not use for private models or personal-memory data. Load only trusted
GitHub Actions checkpoints and restrict torch.load to weights_only=True.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import math

import torch
from tokenizers import Tokenizer

from forge_core.neural.model import ByteTransformer, ModelConfig

REQUIRED = (
    "checkpoint.pt", "metrics.json", "tokenizer.json",
    "source-manifest.json", "train.txt", "valid.txt", "SOURCE-NOTICE.txt",
)


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def audit(folder: Path) -> dict:
    for name in REQUIRED:
        if not (folder / name).is_file():
            raise ValueError(f"Archive missing {name}")
    manifest = json.loads((folder / "source-manifest.json").read_text())
    report = json.loads((folder / "metrics.json").read_text())
    if manifest.get("schema") != 1 or manifest.get("license") != "cdla-sharing-1.0":
        raise ValueError("Source provenance/rights incorrect")
    if manifest.get("revision") != "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64":
        raise ValueError("Source Git revision not approved")
    for part in ("train", "valid"):
        if hash_file(folder / f"{part}.txt") != manifest.get(f"{part}_sha256"):
            raise ValueError(f"{part} data hash mismatch")
    text = (folder / "SOURCE-NOTICE.txt").read_text()
    if "TinyStories" not in text or "CDLA-Sharing-1.0" not in text:
        raise ValueError("Source attribution missing")
    if report.get("source", {}).get("train_sha256") != manifest["train_sha256"]:
        raise ValueError("Metrics refer to different source")
    if report.get("source", {}).get("valid_sha256") != manifest["valid_sha256"]:
        raise ValueError("Metrics refer to different validation")
    if report.get("status") != "research_not_preschool_certified":
        raise ValueError("Unapproved capability claim in model record")
    if report.get("parameters", 0) < 1000 or not math.isfinite(
            float(report.get("validation_token_loss_after", float("nan")))):
        raise ValueError("Invalid metric/parameter record")
    ckpt = torch.load(folder / "checkpoint.pt", map_location="cpu",
                      weights_only=True)
    if ckpt.get("schema") != 3 or ckpt.get("revision") != manifest["revision"]:
        raise ValueError("Checkpoint format/revision mismatch")
    if ckpt.get("train_sha256") != manifest["train_sha256"] or (
            ckpt.get("valid_sha256") != manifest["valid_sha256"]):
        raise ValueError("Checkpoint data provenance mismatch")
    tok_hash = hash_file(folder / "tokenizer.json")
    if ckpt.get("tokenizer_sha256") != tok_hash or report.get("tokenizer_sha256") != tok_hash:
        raise ValueError("Tokenizer content mismatch")
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    config = ModelConfig(**ckpt["configuration"])
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise ValueError("Tokenizer vocabulary does not match model")
    model = ByteTransformer(config)
    model.load_state_dict(ckpt["model"], strict=True)
    if model.num_parameters != report["parameters"] or ckpt.get("step") != report["steps_total"]:
        raise ValueError("Model parameters or optimizer steps inconsistent")
    if "optimizer" not in ckpt or "torch_rng_state" not in ckpt or (
            "sample_rng_state" not in ckpt):
        raise ValueError("Research checkpoint is not resumable")
    if not isinstance(report.get("generation_samples"), list):
        raise ValueError("Missing fixed-seed language samples")
    record = {
        "schema": 1, "result": "internal_integrity_verified_not_language_certified",
        "dataset": manifest["dataset"], "source_revision": manifest["revision"],
        "model_parameters": model.num_parameters, "optimizer_step": ckpt["step"],
        "tokenizer_vocabulary": config.vocab_size,
        "heldout_token_loss": report["validation_token_loss_after"],
        "artifacts_sha256": {name: hash_file(folder / name) for name in REQUIRED},
        "warnings": [
            "Research-only public synthetic stories; not user memories",
            "Source revision pinned, subset is first bounded prefix",
            "No independent preschool or instruction-following certificate",
            "A draft release is not yet public and is not a permanent backup guarantee",
        ],
    }
    (folder / "archive-integrity.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.folder), indent=2))


if __name__ == "__main__":
    main()
