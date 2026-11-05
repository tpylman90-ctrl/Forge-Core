"""Reproducible bounded slices from a PINNED TinyStories git revision.

Streams only the first bounded bytes of each large text file. Uses different
upstream train and validation files; never downloads the 1.9GB training set
in full and never uploads private data. This is a reproducible *prefix*
subset, not a statistically representative full corpus.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

DATASET = "roneneldan/TinyStories"
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
LICENSE = "cdla-sharing-1.0"
TRAIN_FILE = "TinyStories-train.txt"
VALID_FILE = "TinyStories-valid.txt"


def source_url(filename: str) -> str:
    if filename not in (TRAIN_FILE, VALID_FILE):
        raise ValueError("Unapproved source file")
    return f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{filename}"


def fetch_prefix(filename: str, byte_limit: int) -> bytes:
    """Bounded streaming read, including a small overread for a complete story."""
    if not 100_000 <= byte_limit <= 64 * 1024 * 1024:
        raise ValueError("Requested sample outside 64 MiB safety bound")
    cap = byte_limit + 16384
    request = Request(source_url(filename), headers={
        "User-Agent": "Forge-Core-research/0.7 (pinned, bounded public text)",
        "Range": f"bytes=0-{cap - 1}",
    })
    with urlopen(request, timeout=90) as response:
        # Some Xet/CDN endpoints ignore Range; never read the entire huge file.
        data = response.read(cap)
    if len(data) < byte_limit:
        raise ValueError(f"Source {filename} shorter than requested prefix")
    return data[:byte_limit]


def story_hash(text: str) -> str:
    return hashlib.sha256(" ".join(text.casefold().split()).encode("utf-8")).hexdigest()


def complete_stories(data: bytes) -> list[str]:
    text = data.decode("utf-8", errors="ignore").replace("\r\n", "\n")
    # Keep only complete records. A story may contain multiple paragraphs.
    if "<|endoftext|>" in text:
        text = text.rsplit("<|endoftext|>", 1)[0]
        stories = [item.strip() for item in text.split("<|endoftext|>")]
    elif "\n\n" in text:
        text = text.rsplit("\n\n", 1)[0]
        stories = [item.strip() for item in re.split(r"\n\s*\n", text)]
    else:
        raise ValueError("Source prefix contains no recognized document separator")
    stories = [item for item in stories if len(item) > 30]
    if len(stories) < 30:
        raise ValueError("Too few complete source records")
    return stories


def clean_prefix(data: bytes, blocked: set[str] | None = None) -> tuple[str, int]:
    accepted, seen = [], set()
    for item in complete_stories(data):
        key = story_hash(item)
        if key not in seen and (blocked is None or key not in blocked):
            seen.add(key)
            accepted.append(item)
    if len(accepted) < 30:
        raise ValueError("Insufficient independent stories after duplicate screening")
    return "\n\n".join(accepted) + "\n", len(accepted)

def prepare(out: Path, train_bytes: int = 10 * 1024 * 1024,
            valid_bytes: int = 2 * 1024 * 1024) -> dict:
    train_raw = fetch_prefix(TRAIN_FILE, train_bytes)
    valid_raw = fetch_prefix(VALID_FILE, valid_bytes)
    train, train_count = clean_prefix(train_raw)
    train_keys = {story_hash(item) for item in complete_stories(train_raw)}
    valid_before = len(complete_stories(valid_raw))
    valid, valid_count = clean_prefix(valid_raw, blocked=train_keys)
    if hashlib.sha256(train.encode()).hexdigest() == hashlib.sha256(valid.encode()).hexdigest():
        raise ValueError("Train and validation text identical")
    overlap_removed = valid_before - valid_count
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.txt").write_text(train, encoding="utf-8")
    (out / "valid.txt").write_text(valid, encoding="utf-8")
    manifest = {
        "schema": 1,
        "dataset": DATASET,
        "revision": REVISION,
        "license": LICENSE,
        "train_url": source_url(TRAIN_FILE),
        "valid_url": source_url(VALID_FILE),
        "source_train_bytes_requested": train_bytes,
        "source_valid_bytes_requested": valid_bytes,
        "train_document_count": train_count, "valid_document_count": valid_count,
        "validation_records_removed_as_duplicate_or_train_overlap": overlap_removed,
        "train_content_bytes": len(train.encode()), "valid_content_bytes": len(valid.encode()),
        "train_sha256": hashlib.sha256(train.encode()).hexdigest(),
        "valid_sha256": hashlib.sha256(valid.encode()).hexdigest(),
        "limitations": [
            "Only beginning of revision-pinned upstream training/validation files is sampled",
            "Exact duplicate screening only; not complete cross-source contamination audit",
            "Source is synthetic stories; real-world knowledge and instruction following not implied",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/preschool"))
    p.add_argument("--train-mib", type=int, default=10)
    p.add_argument("--valid-mib", type=int, default=2)
    a = p.parse_args()
    print(json.dumps(prepare(a.out, a.train_mib * 1024 * 1024,
                             a.valid_mib * 1024 * 1024), indent=2))


if __name__ == "__main__":
    main()
