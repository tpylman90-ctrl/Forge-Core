"""Evaluate archived language checkpoints on a larger, disjoint TinyStories range.

No optimizer updates are performed. The range begins 24 MiB beyond the pinned
training prefix, is 16 MiB long, excludes exact train/validation/previous-test
story duplicates, and is scored with deterministic uniformly spaced windows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.multilane_v1 import fetch_language_test, sha256_file
from forge_core.neural.seed_v07 import heldout

SEQ_LEN = 128
DEFAULT_WINDOWS = 4096
TOKENIZER_SHA256 = "4ecd7803d4c369e80bda2e74c30cfb769b7e3a942f931df83383eec9f794f04c"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def evaluate_checkpoint(checkpoint_dir: Path, tokens: torch.Tensor,
                        tokenizer_sha256: str, windows: int) -> dict:
    checkpoint_path = checkpoint_dir / "checkpoint.pt"
    tokenizer_path = checkpoint_dir / "tokenizer.json"
    if not checkpoint_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint or tokenizer in {checkpoint_dir}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    actual_tokenizer_sha = sha256_file(tokenizer_path)
    if actual_tokenizer_sha != tokenizer_sha256 or checkpoint.get("tokenizer_sha256") != tokenizer_sha256:
        raise ValueError("Checkpoint tokenizer hash mismatch")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    model = ByteTransformer(ModelConfig(**checkpoint["configuration"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    loss = heldout(model, tokens, windows=windows, seq_len=SEQ_LEN)
    return {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "step": checkpoint.get("step"),
        "loss": round(loss, 8),
    }


def evaluate_pair(source: Path, parent_dir: Path, candidate_dir: Path, out: Path,
                  *, byte_offset: int, sample_bytes: int, windows: int) -> dict:
    source_manifest_path = source / "source-manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("revision") != "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64":
        raise ValueError("TinyStories revision is not the pinned approved revision")
    if source_manifest.get("license") != "cdla-sharing-1.0":
        raise ValueError("Unexpected TinyStories license metadata")
    old_test_path = source / "test.txt"
    old_test_sha = sha256_file(old_test_path)
    if old_test_sha != source_manifest.get("test_sha256"):
        raise ValueError("Existing independent test data does not match its manifest")
    if sha256_file(source / "tokenizer.json") != TOKENIZER_SHA256:
        raise ValueError("Forge tokenizer SHA-256 mismatch")
    for split in ("train", "valid"):
        if sha256_file(source / f"{split}.txt") != source_manifest[f"{split}_sha256"]:
            raise ValueError(f"{split} data hash mismatch")

    test_text, test_manifest = fetch_language_test(
        source, source_manifest, byte_offset=byte_offset,
        sample_bytes=sample_bytes, exclude_paths=(old_test_path,),
    )
    test_path = source / "test.txt"
    test_sha = sha256_file(test_path)
    if test_sha == old_test_sha:
        raise ValueError("New independent test slice matches the earlier test")
    tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    test_tokens = torch.tensor(tokenizer.encode(test_text).ids, dtype=torch.long)
    if test_tokens.numel() <= SEQ_LEN + 1:
        raise ValueError("Larger independent test slice is too small")
    parent = evaluate_checkpoint(parent_dir, test_tokens, TOKENIZER_SHA256, windows)
    candidate = evaluate_checkpoint(candidate_dir, test_tokens, TOKENIZER_SHA256, windows)
    report = {
        "schema": 1,
        "status": "evaluation_only_no_training",
        "dataset": {
            "name": "TinyStories",
            "revision": test_manifest["revision"],
            "license": test_manifest["license"],
            "train_sha256": test_manifest["train_sha256"],
            "validation_sha256": test_manifest["valid_sha256"],
            "previous_test_sha256": old_test_sha,
            "new_test_sha256": test_sha,
            "raw_range_sha256": test_manifest["test_source"]["raw_range_sha256"],
            "raw_range_start_inclusive": test_manifest["test_source"]["range_start_inclusive"],
            "raw_range_end_inclusive": test_manifest["test_source"]["range_end_inclusive"],
            "raw_range_bytes": sample_bytes,
            "offset_after_training_prefix_bytes": byte_offset,
            "utf8_bytes_after_deduplication": test_path.stat().st_size,
            "tokens_after_deduplication": int(test_tokens.numel()),
            "documents_after_deduplication": test_manifest["test_document_count"],
            "exact_duplicates_removed": test_manifest["test_source"]["exact_train_or_validation_duplicates_removed"],
            "previous_test_files_excluded": test_manifest["test_source"]["prior_test_files_excluded"],
            "tokenizer_sha256": TOKENIZER_SHA256,
        },
        "evaluation": {
            "context_tokens": SEQ_LEN,
            "uniform_windows": windows,
            "parent": parent,
            "candidate": candidate,
            "candidate_minus_parent_loss": round(candidate["loss"] - parent["loss"], 8),
            "interpretation": "Fixed larger held-out next-token loss only; this does not demonstrate comprehension or reasoning.",
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "large-independent-test.txt").write_text(test_text, encoding="utf-8")
    (out / "large-independent-test-manifest.json").write_text(
        json.dumps(test_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--parent-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--offset-mib", type=int, default=24)
    parser.add_argument("--range-mib", type=int, default=16)
    parser.add_argument("--windows", type=int, default=DEFAULT_WINDOWS)
    args = parser.parse_args()
    if args.offset_mib < 24 or args.offset_mib % 4 or args.range_mib < 4 or args.range_mib % 4:
        parser.error("offset and range must be 4 MiB multiples; offset must be at least 24 MiB")
    if not 256 <= args.windows <= 16384:
        parser.error("windows must be between 256 and 16384")
    evaluate_pair(
        args.source, args.parent_dir, args.candidate_dir, args.out,
        byte_offset=args.offset_mib * 1024 * 1024,
        sample_bytes=args.range_mib * 1024 * 1024, windows=args.windows,
    )


if __name__ == "__main__":
    main()
