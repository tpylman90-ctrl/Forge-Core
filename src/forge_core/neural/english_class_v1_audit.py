"""Read-only audit of the first Forge English Class experiment.

The first run trained three separate, unpromoted models correctly but
scored FROZEN story loss on 32 validation windows and adapted losses on
48. Those ratios are not comparable. Recompute all four losses over
the exact same 48 windows WITHOUT retraining or touching the models.
Recheck the exam predictions against saved reports for integrity only.
This previously inspected synthetic exam is NOT a fresh held-out test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from forge_core.neural.english_class_v1 import (
    ARMS, comparable_story_loss, make_exercises, score,
)
from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.preschool_qa import verify_source


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_lineage(record: dict, *, arm: str, original_sha: str,
                  tokenizer_sha: str, configuration: dict) -> None:
    """Reject accidental weight substitution before evaluating a model."""
    if (record.get("schema") != 1 or not record.get("research_only")
            or record.get("arm") != arm
            or record.get("source_checkpoint_sha256") != original_sha
            or record.get("source_tokenizer_sha256") != tokenizer_sha
            or record.get("configuration") != configuration
            or "best_weights" not in record):
        raise ValueError(f"Invalid experimental model lineage: {arm}")


def audit(*, source: Path, trained: Path, out: Path) -> dict:
    torch.set_num_threads(min(2, torch.get_num_threads()))
    original, manifest, codec = verify_source(source)
    original_sha = sha_file(source / "checkpoint.pt")
    tok_sha = sha_file(source / "tokenizer.json")
    report_path = trained / "english-class-v1.json"
    if not report_path.is_file():
        raise ValueError("Missing exact original English Class experiment report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("schema") != 1 or
            report.get("status") != "research_only_unpromoted"
            or report.get("source_run_id") != 41572257363
            or report.get("source_checkpoint_sha256") != original_sha
            or report.get("source_tokenizer_sha256") != tok_sha
            or report.get("train_sha256") != manifest["train_sha256"]
            or report.get("valid_sha256") != manifest["valid_sha256"]
            or report.get("seed") != 7331
            or report.get("exercise_counts", {}).get("exam") != 96):
        raise ValueError("Research report fails source and experiment audit")
    exam = make_exercises("exam", per_skill=16, seed=report["seed"] + 97)
    exam_digest = hashlib.sha256("\n".join(
        r.prompt for r in exam).encode()).hexdigest()
    if report["exercise_sha256"]["exam"] != exam_digest:
        raise ValueError("Exam prompts differ from original logged test")
    ids_a, ids_b = codec.encode("A").ids, codec.encode("B").ids
    if len(ids_a) != 1 or len(ids_b) != 1 or ids_a == ids_b:
        raise ValueError("Original tokenizer cannot score atomic A/B labels")
    answer_ids = ids_a[0], ids_b[0]
    val_tokens = torch.tensor(codec.encode(
        (source / "valid.txt").read_text(encoding="utf-8")).ids,
                              dtype=torch.long)
    started = time.monotonic()
    original_model = ByteTransformer(ModelConfig(**original["configuration"]))
    original_model.load_state_dict(original["model"], strict=True)
    original_model.eval()
    baseline = comparable_story_loss(original_model, val_tokens)
    baseline_exam = score(original_model, exam, codec, answer_ids)
    if baseline_exam != report["frozen"]["synthetic_exam"]:
        raise ValueError("Original exam decisions differ from saved baseline")
    verified = {}
    for arm in ARMS:
        path = trained / f"{arm}-research.pt"
        if not path.is_file():
            raise ValueError("Expected an experimental-only checkpoint for all arms")
        record = torch.load(path, map_location="cpu", weights_only=True)
        check_lineage(record, arm=arm, original_sha=original_sha,
                      tokenizer_sha=tok_sha,
                      configuration=original["configuration"])
        model = ByteTransformer(ModelConfig(**original["configuration"]))
        model.load_state_dict(record["best_weights"], strict=True)
        model.eval()
        examined = score(model, exam, codec, answer_ids)
        if examined != report["arms"][arm]["sealed_synthetic_exam"]:
            raise ValueError(f"Saved and rescored answers disagree for {arm}")
        retained = comparable_story_loss(model, val_tokens)
        verified[arm] = {
            "reconfirmed_exam": examined,
            "comparable_story_token_loss": retained,
            "ratio_to_original": round(retained / baseline, 5),
            "within_five_percent": retained <= baseline * 1.05,
            "selected_development_step": record["best_dev_step"],
            "source_sha256_reverified": True,
        }
    output = {
        "schema": 1, "audit_not_retraining": True,
        "no_models_promoted": True,
        "source_run_id": 41572257363,
        "original_english_training_run_id": 41775152111,
        "original_checkpoint_sha256": original_sha,
        "tokenizer_sha256": tok_sha,
        "new_validation_windows": 48,
        "validation_context_length": 64,
        "old_baseline_window_count": 32,
        "old_retention_ratios_invalid_for_comparison": True,
        "exam_is_rechecked_not_newly_untouched": True,
        "base_exam": baseline_exam,
        "original_story_token_loss": baseline,
        "arms": verified,
        "duration_seconds": round(time.monotonic()-started, 2),
        "limitations": [
            "The synthetic English Class exam has been previously inspected",
            "Re-scoring establishes integrity, NOT new language generalization",
            "Story token loss sampled at fixed positions is not free-form fluency",
            "All training models remain experimental and have not been promoted",
            "Android and user memory are separate from this public research",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "english-class-v1-retention-audit.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "base_story_loss": baseline,
        "original_exam_by_skill": {
            skill: values["correct"]
            for skill, values in baseline_exam["by_skill"].items()},
        "corrected": {arm: {
            "exam": row["reconfirmed_exam"]["accuracy"],
            "exam_by_skill": {
                skill: values["correct"]
                for skill, values in row["reconfirmed_exam"]["by_skill"].items()},
            "language_loss_ratio": row["ratio_to_original"],
            "retained_within_five_percent": row["within_five_percent"],
        } for arm, row in verified.items()},
    }, indent=2), flush=True)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--trained", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("results/english-class-v1-audit"))
    args = p.parse_args()
    audit(source=args.source, trained=args.trained, out=args.out)


if __name__ == "__main__":
    main()
