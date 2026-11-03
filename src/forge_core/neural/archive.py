"""Audit an approved, public-data Forge Seed checkpoint before archival.

This checks provenance and internal consistency, not intelligence or security.
Only load checkpoints created by trusted Forge Core training workflows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import torch


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_archive(folder: Path, *, require_public_tinystories: bool = True) -> dict:
    manifest_path = folder / "dataset-manifest.json"
    metrics_path = folder / "metrics.json"
    ckpt_path = folder / "checkpoint.pt"
    train_path = folder / "train.txt"
    valid_path = folder / "valid.txt"
    notice_path = folder / "SOURCE-NOTICE.txt"
    paths = (manifest_path, metrics_path, ckpt_path, train_path, valid_path, notice_path)
    if not all(p.is_file() for p in paths):
        raise ValueError("Archive is missing required model, provenance, or public corpus files")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    row = metrics.get("last_run")
    if not isinstance(row, dict):
        raise ValueError("Missing final training metrics")
    if require_public_tinystories:
        if manifest.get("source") != "roneneldan/TinyStories" or manifest.get("license") != "cdla-sharing-1.0":
            raise ValueError("Only the reviewed public TinyStories source is archive-eligible")
    if manifest.get("schema") != 2 or not manifest.get("train_documents") or not manifest.get("valid_documents"):
        raise ValueError("Invalid document provenance manifest")
    if not notice_path.read_text(encoding="utf-8").strip():
        raise ValueError("Missing source license/attribution notice")
    train_hash, valid_hash = digest(train_path), digest(valid_path)
    if train_hash != manifest.get("train_sha256") or valid_hash != manifest.get("valid_sha256"):
        raise ValueError("Archived corpus differs from recorded provenance")
    if train_hash != row.get("train_sha256") or valid_hash != row.get("valid_sha256"):
        raise ValueError("Metrics refer to a different training corpus")
    if row.get("manifest_sha256") != digest(manifest_path):
        raise ValueError("Manifest fingerprint differs from metrics")
    # weights_only prevents unrestricted Python pickle object reconstruction.
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != 2 or checkpoint.get("manifest_sha256") != digest(manifest_path):
        raise ValueError("Checkpoint provenance mismatch")
    if checkpoint.get("train_sha256") != train_hash or checkpoint.get("valid_sha256") != valid_hash:
        raise ValueError("Checkpoint corpus hashes mismatch")
    if checkpoint.get("step") != row.get("global_step") or checkpoint.get("config") is None:
        raise ValueError("Checkpoint training step or configuration mismatch")
    if "model" not in checkpoint or "optimizer" not in checkpoint or "sample_rng_state" not in checkpoint:
        raise ValueError("Checkpoint is not resumable")
    if not math.isfinite(float(row.get("heldout_loss_after", float("nan")))):
        raise ValueError("Model validation loss is missing or not finite")
    expected_params = row.get("parameter_count")
    actual_params = sum(x.numel() for x in checkpoint["model"].values())
    # Tied weights appear under two names in state_dict; parameter count is
    # measured separately during the experiment, so don't compare directly.
    if not isinstance(expected_params, int) or expected_params < 1 or actual_params < expected_params:
        raise ValueError("Checkpoint parameter metadata invalid")
    record = {
        "schema": 1,
        "source": manifest["source"],
        "license": manifest["license"],
        "model_size": row.get("model_size"),
        "trainable_parameters": expected_params,
        "optimizer_steps": checkpoint["step"],
        "heldout_loss_after": row["heldout_loss_after"],
        "archive_status": "internally_consistent_not_language_certified",
        "checksums": {p.name: digest(p) for p in paths},
    }
    (folder / "archive-audit.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_archive(args.folder), indent=2))


if __name__ == "__main__":
    main()
