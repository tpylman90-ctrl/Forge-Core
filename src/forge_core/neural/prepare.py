"""Prepare tiny, public and explicitly selected training/validation corpora.

This is an educational CPU experiment pipeline, not large-scale pretraining.
'train' and 'validation' are separate source splits. No user memories.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API = "https://datasets-server.huggingface.co/rows"
SOURCE = "roneneldan/TinyStories"
LICENSE = "cdla-sharing-1.0"


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def _get_rows(split: str, count: int) -> list[str]:
    results: list[str] = []
    for offset in range(0, count, 100):
        params = urlencode({
            "dataset": SOURCE, "config": "default", "split": split,
            "offset": offset, "length": min(count - offset, 100),
        })
        request = Request(API + "?" + params, headers={
            "User-Agent": "Forge-Core-research/0.3 (public bounded samples)",
            "Accept": "application/json",
        })
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read(12 * 1024 * 1024).decode("utf-8"))
        for item in payload["rows"]:
            value = item.get("row", {}).get("text")
            if isinstance(value, str):
                results.append(value)
    return results


def _demo(split: str, count: int) -> list[str]:
    # Entirely original tiny fixtures. They are NOT a proper language corpus.
    seeds = (
        "A little fox explored the green valley and found a quiet river.",
        "At sunset the gentle owl carried a small lantern toward the village.",
        "A curious robot opened a wooden door and asked about the bright stars.",
        "The young gardener gave the flowers water before a warm summer rain.",
        "Two friends built a paper boat and watched it float downstream.",
    )
    suffix = " They talked about what they learned and tried again."
    shift = 0 if split == "train" else 2
    return [seeds[(i + shift) % len(seeds)] + suffix + f" Chapter {i + shift}." for i in range(count)]


def clean_unique(items: list[str], seen: set[str]) -> list[str]:
    output: list[str] = []
    for text in items:
        text = normalized(text)
        if not 20 <= len(text) <= 10000:
            continue
        fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if fingerprint not in seen:
            seen.add(fingerprint)
            output.append(text)
    return output


def prepare(source: str, output: Path, train_rows: int = 200, valid_rows: int = 50) -> dict:
    if source not in ("demo", "tinystories"):
        raise ValueError("Unknown data source")
    if not 1 <= train_rows <= 500 or not 1 <= valid_rows <= 200:
        raise ValueError("This CPU experiment allows 1..500 training and 1..200 validation rows")
    origin = _demo if source == "demo" else _get_rows
    seen: set[str] = set()
    # Train first; disallow exact duplicate documents in validation.
    train_items = clean_unique(origin("train", train_rows), seen)
    valid_items = clean_unique(origin("validation", valid_rows), seen)
    if not train_items or not valid_items:
        raise ValueError("The source has no usable non-overlapping train/validation items")
    train_text = "\n\n".join(train_items) + "\n"
    valid_text = "\n\n".join(valid_items) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    (output / "train.txt").write_text(train_text, encoding="utf-8")
    (output / "valid.txt").write_text(valid_text, encoding="utf-8")
    manifest = {
        "schema": 1,
        "source": SOURCE if source == "tinystories" else "Forge Core original CI demo sentences",
        "license": LICENSE if source == "tinystories" else "Forge Core original project fixtures",
        "dataset_url": "https://huggingface.co/datasets/" + SOURCE if source == "tinystories" else None,
        "selection": "first rows of public source train and validation splits; exact duplicates removed",
        "requested_train_rows": train_rows,
        "requested_valid_rows": valid_rows,
        "accepted_train_rows": len(train_items),
        "accepted_valid_rows": len(valid_items),
        "train_sha256": hashlib.sha256(train_text.encode("utf-8")).hexdigest(),
        "valid_sha256": hashlib.sha256(valid_text.encode("utf-8")).hexdigest(),
        "warning": "Small training sample; dataset server content is not revision pinned. Hashes are authoritative.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("demo", "tinystories"), required=True)
    parser.add_argument("--out", type=Path, default=Path("data/experiment"))
    parser.add_argument("--train-rows", type=int, default=200)
    parser.add_argument("--valid-rows", type=int, default=50)
    args = parser.parse_args()
    manifest = prepare(args.source, args.out, args.train_rows, args.valid_rows)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
