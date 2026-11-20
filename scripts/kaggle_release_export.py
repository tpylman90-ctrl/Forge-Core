#!/usr/bin/env python3
"""Gate and store a Forge Core Kaggle candidate in a GitHub draft release.

This script exports research artifacts only. It never publishes a release or
changes a model pointer. The GitHub token must be provided through the
FORGE_GITHUB_TOKEN environment variable (for Kaggle, use Kaggle Secrets).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from datetime import datetime, timezone


REPO_DEFAULT = "tpylman90-ctrl/Forge-Core"
ALLOWED_LANES = {"language", "blender", "godot"}
MIN_RELATIVE_GAIN = 0.002  # 0.2% on both validation and independent test.
MAX_CORE_REGRESSION = 0.05
API = "https://api.github.com"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_child(root: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe bundle path: {relative}")
    path = (root / Path(*rel.parts)).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"Bundle path escapes root: {relative}")
    return path


def request_json(method: str, url: str, token: str, body: dict | None = None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = Request(url, data=data, method=method, headers={
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "Forge-Core-Kaggle-Research-Exporter",
        **({"Content-Type": "application/json"} if data is not None else {}),
    })
    try:
        with urlopen(req, timeout=90) as response:
            payload = response.read()
            return response.status, json.loads(payload) if payload else {}
    except HTTPError as exc:
        # Deliberately omit response body and headers to avoid leaking secrets.
        raise RuntimeError(f"GitHub API request failed with HTTP {exc.code}: {method} {url.split('?')[0]}") from None
    except URLError as exc:
        raise RuntimeError(f"GitHub API request failed: {exc.reason}") from None


def upload_asset(upload_url: str, file_path: Path, token: str) -> dict:
    url = upload_url.split("{", 1)[0] + "?" + urlencode({"name": file_path.name})
    req = Request(url, data=file_path.read_bytes(), method="POST", headers={
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/octet-stream",
        "User-Agent": "Forge-Core-Kaggle-Research-Exporter",
    })
    try:
        with urlopen(req, timeout=300) as response:
            return json.loads(response.read())
    except HTTPError as exc:
        raise RuntimeError(f"GitHub asset upload failed with HTTP {exc.code}; token and bundle were not printed") from None
    except URLError as exc:
        raise RuntimeError(f"GitHub asset upload failed: {exc.reason}") from None


def qualify(root: Path, manifest_path: Path) -> tuple[dict, dict, dict[str, Path]]:
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    if meta.get("schema") != 1 or meta.get("repository") != REPO_DEFAULT:
        raise ValueError("Export manifest schema or repository is not approved")
    lane = meta.get("lane")
    if lane not in ALLOWED_LANES:
        raise ValueError(f"Unsupported lane: {lane!r}; visual runs need their own validated gate")
    if not re.fullmatch(r"[0-9a-f]{40}", str(meta.get("source_commit", ""))):
        raise ValueError("A full pinned Forge source commit is required")
    if not re.fullmatch(r"[0-9a-f]{64}", str(meta.get("input_bundle_sha256", ""))):
        raise ValueError("A SHA-256 fingerprint of the attached input bundle is required")
    sources = meta.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("List each training source, revision, license, and use review")
    for source in sources:
        if not all(source.get(k) for k in ("name", "url", "revision", "license")):
            raise ValueError("Training source metadata is incomplete")
        if source.get("training_allowed") is not True or source.get("weight_distribution_allowed") is not True:
            raise ValueError("Source rights do not explicitly allow training and checkpoint distribution")
    if meta.get("private_data_in_training") is not False:
        raise ValueError("Private or unverified training data cannot be exported")

    paths = {key: safe_child(root, meta[key]) for key in (
        "metrics_path", "lineage_path", "checkpoint_path", "parent_checkpoint_path",
        "tokenizer_path", "source_manifest_path")}
    if any(not p.is_file() for p in paths.values()):
        raise ValueError("Export bundle is missing checkpoint, lineage, tokenizer, metrics, or source manifest")
    metrics = json.loads(paths["metrics_path"].read_text(encoding="utf-8"))
    lineage = json.loads(paths["lineage_path"].read_text(encoding="utf-8"))
    source_manifest = json.loads(paths["source_manifest_path"].read_text(encoding="utf-8"))

    if metrics.get("status") != "research_only_not_promoted" or metrics.get("lane") != lane:
        raise ValueError("Candidate status or lane does not match the export manifest")
    seed = metrics.get("seed")
    if not isinstance(seed, int) or seed not in (43, 53, 73) or seed != meta.get("seed"):
        raise ValueError("Candidate seed mismatch")
    if metrics.get("weights_changed_from_parent") is not True or lineage.get("weights_changed_from_parent") is not True:
        raise ValueError("Candidate weights are unchanged; there is nothing useful to export")
    if lineage.get("personal_memory_in_weights") is not False or lineage.get("isolated_lane_weights") is not True:
        raise ValueError("Checkpoint does not prove personal-data exclusion and lane isolation")
    if metrics.get("optimizer_state_resumed") is not True:
        raise ValueError("Resumable optimizer lineage is required")

    candidate_sha = sha256(paths["checkpoint_path"])
    parent_sha = sha256(paths["parent_checkpoint_path"])
    if candidate_sha != metrics.get("candidate_checkpoint_sha256") or candidate_sha != lineage.get("candidate_sha256"):
        raise ValueError("Candidate checkpoint checksum mismatch")
    if parent_sha != metrics.get("parent_checkpoint_sha256") or parent_sha != lineage.get("parent_sha256"):
        raise ValueError("Parent checkpoint checksum mismatch")
    if sha256(paths["tokenizer_path"]) != metrics.get("tokenizer_sha256"):
        raise ValueError("Tokenizer checksum mismatch")
    if source_manifest.get("revision") is None or source_manifest.get("license") is None:
        raise ValueError("Pinned source revision and license are required")
    if not metrics.get("test_sha256") or metrics.get("test_sha256") in (metrics.get("train_sha256"), metrics.get("valid_sha256")):
        raise ValueError("Independent held-out test split is missing or overlaps train/validation")

    val_parent = float(metrics["domain_validation_loss_parent"])
    val_candidate = float(metrics["domain_validation_loss_candidate"])
    test_parent = float(metrics["domain_test_loss_parent"])
    test_candidate = float(metrics["domain_test_loss_candidate"])
    core_delta = float(metrics["core_language_delta_candidate_minus_parent"])
    if not all(map(lambda x: x >= 0 and x < float("inf"), (val_parent, val_candidate, test_parent, test_candidate))) or not (-float("inf") < core_delta < float("inf")):
        raise ValueError("Evaluation values must be finite and non-negative")
    val_gain = (val_parent - val_candidate) / max(val_parent, 1e-12)
    test_gain = (test_parent - test_candidate) / max(test_parent, 1e-12)
    if val_gain < MIN_RELATIVE_GAIN or test_gain < MIN_RELATIVE_GAIN:
        raise ValueError("No export: candidate must improve both fixed validation and independent test loss by at least 0.2%")
    if metrics.get("core_language_guard_passed") is not True or core_delta > MAX_CORE_REGRESSION:
        raise ValueError("No export: the core-language retention guard failed")
    if meta.get("human_task_regression") is not False:
        raise ValueError("No export: an explicit no-regression result for available task checks is required")

    decision = {
        "schema": 1,
        "eligible_for_draft_archive": True,
        "status": "research_only_unpromoted",
        "lane": lane,
        "seed": seed,
        "source_commit": meta["source_commit"],
        "input_bundle_sha256": meta["input_bundle_sha256"],
        "parent_checkpoint_sha256": parent_sha,
        "candidate_checkpoint_sha256": candidate_sha,
        "domain_validation_loss_parent": val_parent,
        "domain_validation_loss_candidate": val_candidate,
        "domain_validation_relative_gain": round(val_gain, 6),
        "domain_independent_test_loss_parent": test_parent,
        "domain_independent_test_loss_candidate": test_candidate,
        "domain_test_relative_gain": round(test_gain, 6),
        "core_language_delta_candidate_minus_parent": core_delta,
        "core_language_guard_passed": True,
        "promotion": "none; draft storage does not authorize adoption",
    }
    return meta, decision, paths


def make_bundle(decision: dict, meta: dict, paths: dict[str, Path], dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for name, source in paths.items():
        shutil.copy2(source, dest / ("candidate-checkpoint.pt" if name == "checkpoint_path" else
                                     "parent-checkpoint.pt" if name == "parent_checkpoint_path" else
                                     "metrics.json" if name == "metrics_path" else
                                     "lineage.json" if name == "lineage_path" else
                                     "tokenizer.json" if name == "tokenizer_path" else
                                     "source-manifest.json"))
    public_meta = {k: v for k, v in meta.items() if k not in {"token", "github_token", "secret"}}
    (dest / "export-manifest.json").write_text(json.dumps(public_meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (dest / "qualification.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hashes = {p.name: sha256(p) for p in sorted(dest.iterdir()) if p.is_file()}
    (dest / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    archive = dest.parent / "forge-kaggle-candidate.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        for path in sorted(dest.iterdir()):
            if path.is_file():
                tf.add(path, arcname=f"forge-kaggle-candidate/{path.name}")
    return archive


def export(args) -> None:
    token = os.environ.get("FORGE_GITHUB_TOKEN", "")
    if not token:
        raise ValueError("Set FORGE_GITHUB_TOKEN in Kaggle Secrets; never paste it into the notebook")
    root = args.bundle_dir.resolve()
    manifest_path = safe_child(root, args.export_manifest)
    meta, decision, paths = qualify(root, manifest_path)
    run_id = str(meta.get("run_id", ""))
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,60}", run_id):
        raise ValueError("run_id must be a short safe label")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"forge-kaggle-{decision['lane']}-s{decision['seed']}-{run_id}-{stamp}".lower()
    repo = meta["repository"]
    output = args.output.resolve()
    archive = make_bundle(decision, meta, paths, output / "bundle")
    status, release = request_json("POST", f"{API}/repos/{repo}/releases", token, {
        "tag_name": tag,
        "name": f"Forge {decision['lane']} candidate {run_id} (research draft)",
        "draft": True,
        "prerelease": True,
        "body": ("Qualified for research storage by held-out loss and retention gates. "
                 "This remains an unpublished draft; it is not an approved model promotion. "
                 "Review qualification.json, provenance, checksums, and checkpoint lineage before use."),
    })
    upload_asset(release["upload_url"], archive, token)
    print(json.dumps({"result": "draft_release_created", "lane": decision["lane"],
                      "seed": decision["seed"], "tag": tag,
                      "release_url": release.get("html_url"),
                      "asset": archive.name,
                      "domain_test_relative_gain": decision["domain_test_relative_gain"],
                      "promoted": False}, indent=2))


def download(args) -> None:
    """Fetch a reviewed-by-tag draft asset so the same lane can resume it."""
    token = os.environ.get("FORGE_GITHUB_TOKEN", "")
    if not token:
        raise ValueError("Set FORGE_GITHUB_TOKEN in the runtime secret store; never paste it into a notebook")
    repo = args.repository
    if repo != REPO_DEFAULT:
        raise ValueError("Only the configured Forge Core repository is supported")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", args.tag):
        raise ValueError("Invalid release tag")
    _, release = request_json("GET", f"{API}/repos/{repo}/releases/tags/{quote(args.tag, safe='')}", token)
    if release.get("draft") is not True:
        raise ValueError("This helper only retrieves unpublished research drafts")
    assets = [a for a in release.get("assets", []) if a.get("name") == "forge-kaggle-candidate.tar.gz"]
    if len(assets) != 1:
        raise ValueError("Draft must contain exactly one Forge candidate bundle")
    asset = assets[0]
    req = Request(asset["url"], headers={
        "Accept": "application/octet-stream",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "Forge-Core-Kaggle-Research-Exporter",
    })
    dest = args.output.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / asset["name"]
    try:
        with urlopen(req, timeout=300) as response:
            archive.write_bytes(response.read())
    except HTTPError as exc:
        raise RuntimeError(f"GitHub draft asset download failed with HTTP {exc.code}; credentials were not printed") from None
    except URLError as exc:
        raise RuntimeError(f"GitHub draft asset download failed: {exc.reason}") from None

    extracted = dest / "candidate"
    extracted.mkdir(exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        members = tf.getmembers()
        expected_prefix = "forge-kaggle-candidate/"
        for member in members:
            p = PurePosixPath(member.name)
            if not member.isfile() or p.is_absolute() or ".." in p.parts or not member.name.startswith(expected_prefix):
                raise ValueError("Draft archive contains an unsafe or unexpected member")
        for member in members:
            rel = PurePosixPath(member.name)
            target = (extracted / Path(*rel.parts)).resolve()
            if extracted.resolve() not in target.parents:
                raise ValueError("Draft archive member escapes the extraction directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
    payload = extracted / "forge-kaggle-candidate"
    checksums = json.loads((payload / "SHA256SUMS.json").read_text(encoding="utf-8"))
    for name, expected in checksums.items():
        rel = PurePosixPath(name)
        path = (payload / Path(*rel.parts)).resolve()
        if rel.is_absolute() or ".." in rel.parts or payload.resolve() not in path.parents:
            raise ValueError("Checksum manifest contains an unsafe path")
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Downloaded candidate checksum mismatch: {name}")
    qualification = json.loads((payload / "qualification.json").read_text(encoding="utf-8"))
    if qualification.get("eligible_for_draft_archive") is not True or qualification.get("status") != "research_only_unpromoted":
        raise ValueError("Draft qualification metadata is missing or not eligible")
    print(json.dumps({"result": "draft_candidate_downloaded", "tag": args.tag,
                      "lane": qualification["lane"], "seed": qualification["seed"],
                      "directory": str(payload), "verified_files": len(checksums),
                      "promotion": "none"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export", help="qualify and upload a candidate as a GitHub draft release")
    p.add_argument("--bundle-dir", type=Path, required=True)
    p.add_argument("--export-manifest", default="export-manifest.json")
    p.add_argument("--output", type=Path, default=Path("/kaggle/working/forge-github-export"))
    p.set_defaults(func=export)
    d = sub.add_parser("download", help="download a draft candidate by exact release tag and verify its hashes")
    d.add_argument("--repository", default=REPO_DEFAULT)
    d.add_argument("--tag", required=True)
    d.add_argument("--output", type=Path, default=Path("/kaggle/working/forge-github-import"))
    d.set_defaults(func=download)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
