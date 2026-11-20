"""Matched direct-teaching and Ask-Why CPU study for Forge's three lanes."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import torch
from tokenizers import Tokenizer
from forge_core.neural import multilane_v3 as trainer

PACK_ROOT = Path("data/chatgpt_teacher_v1")
PACK_ROOT_V2 = Path("data/chatgpt_teacher_v2")
LANES = ("language", "blender", "godot")
ARMS = ("control", "chatgpt_direct", "chatgpt_askwhy")

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def pack_path(lane: str, curriculum: str, split: str) -> Path:
    if curriculum == "direct":
        filename = "train.txt" if split == "train" else "eval.txt"
    elif curriculum in {"askwhy", "askwhy_v2"}:
        filename = "why_train.txt" if split == "train" else "why_eval.txt"
    else:
        raise ValueError(f"Unknown curriculum: {curriculum}")
    root = PACK_ROOT_V2 if curriculum == "askwhy_v2" else PACK_ROOT
    return root / lane / filename

def stream(lane: str, tokenizer: Tokenizer, curriculum: str) -> torch.Tensor:
    path = pack_path(lane, curriculum, "train")
    ids = tokenizer.encode(path.read_text(encoding="utf-8")).ids
    if len(ids) <= trainer.SEQ_LEN + 1:
        raise ValueError(f"Teacher {lane}/{curriculum} pack is too small")
    return torch.tensor(ids, dtype=torch.long)

def mixed_stream(original: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    # Give the selected lesson stream an equal token share for this controlled arm.
    repeats = max(1, math.ceil(original.numel() / teacher.numel()))
    return torch.cat((original, teacher.repeat(repeats)))

def teacher_manifest(manifest: dict, lane: str, curriculum: str) -> dict:
    result = dict(manifest)
    train_path = pack_path(lane, curriculum, "train")
    eval_path = pack_path(lane, curriculum, "eval")
    result["chatgpt_teacher"] = {
        "teacher": "ChatGPT", "lane": lane, "curriculum": curriculum,
        "pack_version": "chatgpt-tutor-v2" if curriculum == "askwhy_v2" else "askwhy-v1", "license": "CC0-1.0",
        "train_sha256": sha256(train_path.read_bytes()),
        "eval_sha256": sha256(eval_path.read_bytes()),
        "ask_why_protocol": "Forge asks why; teacher explains in short steps with an analogy; Forge restates, applies, and checks",
        "mix_policy": "equal token count of selected teacher lessons and original lane stream",
        "promotion_policy": "research-only; no automatic promotion or cross-lane weight merge",
    }
    return result

def install_treatment(lane: str, tokenizer: Tokenizer, curriculum: str) -> None:
    original_language = trainer.load_language_data
    original_manual = trainer.load_manual_data
    def language_data(source: Path, *args, **kwargs):
        tok_sha, train, valid, test, manifest = original_language(source, *args, **kwargs)
        selected = stream(lane, tokenizer, curriculum)
        return tok_sha, mixed_stream(train, selected), valid, test, teacher_manifest(manifest, lane, curriculum)
    def manual_data(which_lane: str, folder: Path, which_tokenizer: Tokenizer):
        train, valid, test, manifest = original_manual(which_lane, folder, which_tokenizer)
        selected = stream(which_lane, which_tokenizer, curriculum)
        return mixed_stream(train, selected), valid, test, teacher_manifest(manifest, which_lane, curriculum)
    if lane == "language":
        trainer.load_language_data = language_data
    else:
        trainer.load_manual_data = manual_data

@torch.no_grad()
def text_loss(checkpoint: Path, tokenizer_path: Path, text: str) -> float:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model = trainer.ByteTransformer(trainer.ModelConfig(**ckpt["configuration"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    torch.set_num_threads(min(2, torch.get_num_threads()))
    ids = torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)
    windows = max(1, min(64, (ids.numel() - trainer.SEQ_LEN - 1) // trainer.SEQ_LEN))
    return float(trainer.heldout(model, ids, windows=windows, seq_len=trainer.SEQ_LEN))

def evaluate_pack(checkpoint: Path, tokenizer: Path, lane: str, curriculum: str) -> float:
    text = pack_path(lane, curriculum, "eval").read_text(encoding="utf-8")
    return text_loss(checkpoint, tokenizer, text)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=LANES, required=True)
    parser.add_argument("--arm", choices=(*ARMS, "chatgpt_tutor_v2"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--independent-sampling", action="store_true", help="Use a seed-specific sampling stream for a language-lane replication")
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.source / "tokenizer.json"))
    curriculum = {"chatgpt_direct": "direct", "chatgpt_askwhy": "askwhy", "chatgpt_tutor_v2": "askwhy_v2"}.get(args.arm)
    if curriculum:
        install_treatment(args.lane, tokenizer, curriculum)
    report = trainer.run_lane(
        args.lane, args.source, args.out, seconds=900, max_updates=args.updates,
        seed_override=args.seed, eval_every=100, patience=3,
        core_replay_probability=0.25 if args.lane != "language" else 0.0,
        max_core_regression=0.05,
        independent_sampling=args.independent_sampling,
    )
    parent_path = args.source / "checkpoint.pt"
    candidate_path = args.out / "checkpoint.pt"
    direct_parent = evaluate_pack(parent_path, args.source / "tokenizer.json", args.lane, "direct")
    direct_candidate = evaluate_pack(candidate_path, args.out / "tokenizer.json", args.lane, "direct")
    why_parent = evaluate_pack(parent_path, args.source / "tokenizer.json", args.lane, "askwhy")
    why_candidate = evaluate_pack(candidate_path, args.out / "tokenizer.json", args.lane, "askwhy")
    v2_parent = evaluate_pack(parent_path, args.source / "tokenizer.json", args.lane, "askwhy_v2")
    v2_candidate = evaluate_pack(candidate_path, args.out / "tokenizer.json", args.lane, "askwhy_v2")
    study = {
        "arm": args.arm, "lane": args.lane, "seed": args.seed,
        "direct_eval_loss_parent": round(direct_parent, 8),
        "direct_eval_loss_candidate": round(direct_candidate, 8),
        "direct_eval_improvement": round(direct_parent - direct_candidate, 8),
        "askwhy_eval_loss_parent": round(why_parent, 8),
        "askwhy_eval_loss_candidate": round(why_candidate, 8),
        "askwhy_eval_improvement": round(why_parent - why_candidate, 8),
        "tutor_v2_eval_loss_parent": round(v2_parent, 8),
        "tutor_v2_eval_loss_candidate": round(v2_candidate, 8),
        "tutor_v2_eval_improvement": round(v2_parent - v2_candidate, 8),
        "measure": "held-out next-token loss on direct, Ask-Why v1, and tutor-v2 lesson packs; not capability certification",
        "promotion_eligible": False,
        "teacher_manifest": None if curriculum is None else teacher_manifest({}, args.lane, curriculum)["chatgpt_teacher"],
    }
    report["chatgpt_askwhy_study"] = study
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.out / "askwhy-evaluation.json").write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(study, indent=2), flush=True)

if __name__ == "__main__":
    main()
