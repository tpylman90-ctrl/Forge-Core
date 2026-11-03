"""Prepare bounded, document-separated public language experiments.

A dataset-viewer sample is a *development subset*, not a frozen full corpus.
No user memories or private data should enter these experiments.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
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
MAX_TRAIN_ROWS = 1200
MAX_VALID_ROWS = 300


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def shingles(text: str) -> set[tuple[str, ...]]:
    """Word 5-grams; approximate duplicate detection, not a formal audit."""
    words = re.findall(r"\w+", text.casefold())
    return set(zip(*(words[i:] for i in range(5)))) if len(words) >= 5 else {tuple(words)}


def _get_rows(split: str, count: int) -> list[Document]:
    results: list[Document] = []
    for offset in range(0, count, 100):
        params = urlencode({
            "dataset": SOURCE, "config": "default", "split": split,
            "offset": offset, "length": min(count - offset, 100),
        })
        request = Request(API + "?" + params, headers={
            "User-Agent": "Forge-Core-research/0.4 (bounded public examples)",
            "Accept": "application/json",
        })
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read(12 * 1024 * 1024).decode("utf-8"))
        for index, item in enumerate(payload["rows"]):
            value = item.get("row", {}).get("text")
            if isinstance(value, str):
                results.append(Document(f"{split}:{item.get('row_idx', offset + index)}", value))
    return results


def _demo(split: str, count: int) -> list[Document]:
    # Distinct synthetic patterns in each split; this remains a CI fixture only.
    seeds = (
        "A little fox explored the green valley and found a quiet river.",
        "At sunset the gentle owl carried a small lantern toward the village.",
        "A curious robot opened a wooden door and asked about the bright stars.",
        "The young gardener gave the flowers water before a warm summer rain.",
        "Two friends built a paper boat and watched it float downstream.",
    ) if split == "train" else (
        "A patient sailor checked the compass before leaving the harbor.",
        "The visiting musician practiced a melody at the old train station.",
        "A mountain guide studied the clouds and chose a safer trail.",
        "The small library welcomed travelers looking for new stories.",
        "A careful builder measured the wooden bridge before repairing it.",
    )
    return [Document(f"{split}:demo:{i}", seeds[i % len(seeds)] +
                     f" The story ended at chapter {i + 1}.") for i in range(count)]


def clean_documents(items: list[Document], seen_hashes: set[str],
                    seen_shingles: list[set[tuple[str, ...]]]) -> tuple[list[Document], dict]:
    accepted: list[Document] = []
    counts = {"too_short_or_long": 0, "exact_duplicate": 0, "near_duplicate": 0}
    for item in items:
        txt = normalized(item.text)
        if not 40 <= len(txt) <= 20000:
            counts["too_short_or_long"] += 1
            continue
        digest = hashlib.sha256(txt.casefold().encode("utf-8")).hexdigest()
        if digest in seen_hashes:
            counts["exact_duplicate"] += 1
            continue
        sig = shingles(txt)
        # Bounded public previews permit direct Jaccard screening of documents.
        if any(len(sig & other) / len(sig | other) >= 0.88 for other in seen_shingles):
            counts["near_duplicate"] += 1
            continue
        seen_hashes.add(digest)
        seen_shingles.append(sig)
        accepted.append(Document(item.document_id, txt))
    return accepted, counts


def prepare(source: str, output: Path, train_rows: int = 200, valid_rows: int = 50) -> dict:
    if source not in ("demo", "tinystories"):
        raise ValueError("Unknown data source")
    if not (1 <= train_rows <= MAX_TRAIN_ROWS and 1 <= valid_rows <= MAX_VALID_ROWS):
        raise ValueError("Requested row count exceeds the bounded experiment limit")
    origin = _demo if source == "demo" else _get_rows
    seen_hashes: set[str] = set()
    seen_shingles: list[set[tuple[str, ...]]] = []
    train_items, train_dropped = clean_documents(origin("train", train_rows), seen_hashes, seen_shingles)
    valid_items, valid_dropped = clean_documents(origin("validation", valid_rows), seen_hashes, seen_shingles)
    if len(train_items) < 2 or len(valid_items) < 2:
        raise ValueError("Not enough independent train and validation documents")
    train_text = "\n\n".join(doc.text for doc in train_items) + "\n"
    valid_text = "\n\n".join(doc.text for doc in valid_items) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    (output / "train.txt").write_text(train_text, encoding="utf-8")
    (output / "valid.txt").write_text(valid_text, encoding="utf-8")
    manifest = {
        "schema": 2,
        "source": SOURCE if source == "tinystories" else "Forge Core original CI demo sentences",
        "license": LICENSE if source == "tinystories" else "Project-owned synthetic test fixtures",
        "dataset_url": f"https://huggingface.co/datasets/{SOURCE}" if source == "tinystories" else None,
        "source_revision": "unverified: public dataset-viewer snapshot" if source == "tinystories" else "built-in:0.4",
        "selection": "bounded ordered rows; split-specific document IDs; exact + 5-gram near-duplicate filtering",
        "requested_train_rows": train_rows, "requested_valid_rows": valid_rows,
        "accepted_train_rows": len(train_items), "accepted_valid_rows": len(valid_items),
        "rejected_train": train_dropped, "rejected_valid": valid_dropped,
        "train_documents": [{"id": d.document_id, "sha256": hashlib.sha256(d.text.encode()).hexdigest()}
                            for d in train_items],
        "valid_documents": [{"id": d.document_id, "sha256": hashlib.sha256(d.text.encode()).hexdigest()}
                            for d in valid_items],
        "train_sha256": hashlib.sha256(train_text.encode()).hexdigest(),
        "valid_sha256": hashlib.sha256(valid_text.encode()).hexdigest(),
        "warning": "Small, unfrozen public preview when using TinyStories; not a permanent pretraining corpus. "
                   "Source revisions and full-document leakage still need formal verification.",
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
    info = prepare(args.source, args.out, args.train_rows, args.valid_rows)
    print(json.dumps({k: v for k, v in info.items() if k not in ("train_documents", "valid_documents")}, indent=2))


if __name__ == "__main__":
    main()
