"""Exact-content locks for bounded public language experiments.

This *content* lock is not an upstream Git revision or dataset redistribution.
It fails closed if the remote preview is reordered or modified. A separate
immutable source snapshot is a future milestone for full-corpus training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify_corpus_lock(manifest_path: Path, lock_path: Path) -> dict:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    data = manifest_path.read_bytes()
    manifest = json.loads(data)
    if lock.get("schema") != 1 or manifest.get("schema") != 2:
        raise ValueError("Unsupported dataset lock or manifest version")
    expected = {
        "source": lock.get("dataset"),
        "license": lock.get("license"),
        "requested_train_rows": lock.get("train_rows"),
        "requested_valid_rows": lock.get("valid_rows"),
        "train_sha256": lock.get("train_sha256"),
        "valid_sha256": lock.get("valid_sha256"),
    }
    mismatch = [key for key, value in expected.items()
                if value is None or manifest.get(key) != value]
    if mismatch:
        raise ValueError("Dataset lock mismatch: " + ", ".join(mismatch))
    digest = hashlib.sha256(data).hexdigest()
    if digest != lock.get("manifest_sha256"):
        raise ValueError("Dataset manifest fingerprint differs from locked experiment")
    train_ids = {row["id"] for row in manifest.get("train_documents", [])}
    valid_ids = {row["id"] for row in manifest.get("valid_documents", [])}
    if not train_ids or not valid_ids or train_ids.intersection(valid_ids):
        raise ValueError("Training and validation document IDs overlap or are missing")
    if len(train_ids) != len(manifest["train_documents"]) or len(valid_ids) != len(manifest["valid_documents"]):
        raise ValueError("Duplicate document IDs")
    return {"lock": lock["name"], "manifest_sha256": digest, "status": "verified",
            "limitation": "Content locked, upstream dataset revision not immutable"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_corpus_lock(args.manifest, args.lock), indent=2))


if __name__ == "__main__":
    main()
