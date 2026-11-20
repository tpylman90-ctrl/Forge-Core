"""Controlled ChatGPT-authored lesson pilot for Forge's isolated CPU lanes."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import torch
from tokenizers import Tokenizer
from forge_core.neural import multilane_v3 as trainer

PACK_ROOT = Path("data/chatgpt_teacher_v1")
LANES = ("language", "blender", "godot")

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def teacher_stream(lane: str, tokenizer: Tokenizer, split: str) -> torch.Tensor:
    path = PACK_ROOT / lane / f"{split}.txt"
    ids = tokenizer.encode(path.read_text(encoding="utf-8")).ids
    if len(ids) <= trainer.SEQ_LEN + 1:
        raise ValueError(f"Teacher {lane}/{split} pack is too small")
    return torch.tensor(ids, dtype=torch.long)

def mixed_stream(original: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    target_teacher = math.ceil(original.numel())
    repeats = max(1, math.ceil(target_teacher / teacher.numel()))
    return torch.cat((original, teacher.repeat(repeats)))

def teacher_manifest(manifest: dict, lane: str) -> dict:
    result = dict(manifest)
    train = PACK_ROOT / lane / "train.txt"
    exam = PACK_ROOT / lane / "eval.txt"
    result["chatgpt_teacher"] = {
        "teacher": "ChatGPT", "lane": lane, "pack_version": "v1",
        "license": "CC0-1.0", "train_sha256": sha256(train.read_bytes()),
        "eval_sha256": sha256(exam.read_bytes()),
        "mix_policy": "equal token count of teacher examples and original lane stream",
        "promotion_policy": "research-only; no automatic promotion or cross-lane weight merge",
    }
    return result

def install_treatment(lane: str, tokenizer: Tokenizer) -> None:
    original_language = trainer.load_language_data
    original_manual = trainer.load_manual_data
    def language_data(source: Path, *args, **kwargs):
        tok_sha, train, valid, test, manifest = original_language(source, *args, **kwargs)
        return tok_sha, mixed_stream(train, teacher_stream(lane, tokenizer, "train")), valid, test, teacher_manifest(manifest, lane)
    def manual_data(which_lane: str, folder: Path, which_tokenizer: Tokenizer):
        train, valid, test, manifest = original_manual(which_lane, folder, which_tokenizer)
        result = torch.cat((train, teacher_stream(which_lane, which_tokenizer, "train").repeat(
            max(1, math.ceil(train.numel() / teacher_stream(which_lane, which_tokenizer, "train").numel())))))
        return result, valid, test, teacher_manifest(manifest, which_lane)
    if lane == "language":
        trainer.load_language_data = language_data
    else:
        trainer.load_manual_data = manual_data

@torch.no_grad()
def teacher_loss(checkpoint: Path, tokenizer_path: Path, text: str) -> float:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model = trainer.ByteTransformer(trainer.ModelConfig(**ckpt["configuration"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    torch.set_num_threads(min(2, torch.get_num_threads()))
    ids = torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)
    windows = max(1, min(64, (ids.numel() - trainer.SEQ_LEN - 1) // trainer.SEQ_LEN))
    return float(trainer.heldout(model, ids, windows=windows, seq_len=trainer.SEQ_LEN))

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=LANES, required=True)
    parser.add_argument("--arm", choices=("control", "chatgpt"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=500)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.source / "tokenizer.json"))
    if args.arm == "chatgpt":
        install_treatment(args.lane, tokenizer)
    report = trainer.run_lane(
        args.lane, args.source, args.out, seconds=900, max_updates=args.updates,
        seed_override=args.seed, eval_every=100, patience=3,
        core_replay_probability=0.25 if args.lane != "language" else 0.0,
        max_core_regression=0.05,
    )
    eval_text = (PACK_ROOT / args.lane / "eval.txt").read_text(encoding="utf-8")
    parent_loss = teacher_loss(args.source / "checkpoint.pt", args.source / "tokenizer.json", eval_text)
    candidate_loss = teacher_loss(args.out / "checkpoint.pt", args.out / "tokenizer.json", eval_text)
    study = {
        "arm": args.arm, "lane": args.lane, "seed": args.seed,
        "teacher_eval_loss_parent": round(parent_loss, 8),
        "teacher_eval_loss_candidate": round(candidate_loss, 8),
        "teacher_eval_improvement": round(parent_loss - candidate_loss, 8),
        "measure": "held-out next-token loss; not capability certification",
        "promotion_eligible": False,
        "teacher_manifest": teacher_manifest({}, args.lane)["chatgpt_teacher"],
    }
    report["chatgpt_teacher_study"] = study
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.out / "teacher-evaluation.json").write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(study, indent=2), flush=True)

if __name__ == "__main__":
    main()
