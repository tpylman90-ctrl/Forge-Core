"""Bounded, matched Qwen3-to-Forge language sequence-distillation study.

Qwen is inference-only. Forge starts from its verified scratch-origin 95k
checkpoint and keeps its own tokenizer, architecture, and neural weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import torch
from tokenizers import Tokenizer

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.multilane_v1 import load_language_data
from forge_core.neural.seed_v07 import heldout
from forge_core.neural.train import make_batch

QWEN_REPO = "Qwen/Qwen3-0.6B"
QWEN_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
QWEN_LICENSE = "Apache-2.0"
PARENT_RUN = 42143390701
PARENT_STEP = 95000
PARENT_SHA256 = "a7bcdb996e440f44d7496b887b54d19324927b4296fe9ed2f0f0293c88c63a0a"
TOKENIZER_SHA256 = "4ecd7803d4c369e80bda2e74c30cfb769b7e3a942f931df83383eec9f794f04c"
SEQ_LEN = 128
BATCH = 4
EVAL_WINDOWS = 2048

TOPICS_TRAIN = (
    "a child building a small boat from paper", "a dog waiting beside a garden gate",
    "two friends repairing a kite", "a lantern glowing in a quiet cabin",
    "a rabbit discovering a path through tall grass", "a family preparing a picnic",
    "a bird carrying a ribbon to its nest", "a rainy afternoon at a library",
    "a child learning to ride a bicycle", "a baker sharing warm bread",
    "a turtle crossing a sunny courtyard", "a lost button found under a chair",
    "a group planting seeds in a window box", "a fox listening near a stream",
    "a handmade map with a missing corner", "a train arriving at a tiny station",
    "a patient gardener caring for a young tree", "a blue scarf caught in a breeze",
    "a quiet workshop full of wooden toys", "a penguin following shiny footprints",
    "a neighbor returning a borrowed book", "a small boat rocking beside a dock",
    "a curious mouse exploring a music room", "a child sharing an umbrella",
)
TOPICS_VALID = (
    "a squirrel finding a smooth stone", "a paper crown on a kitchen table",
    "a pair of boots drying by the door", "a sleepy cat watching snow",
    "a friend leaving a kind note", "a red wagon rolling down a gentle hill",
    "a tiny pond behind an old barn", "a child learning a simple song",
)
STYLES = (
    "Tell a gentle story in three short sentences.",
    "Describe the scene with clear, varied sentences.",
    "Write a short passage with a beginning and a small change.",
    "Use one sentence with a cause and its result.",
    "Write a brief dialogue followed by one sentence of narration.",
    "Explain what happens in plain, natural English.",
    "Write a short passage using a pronoun that clearly refers to a person or thing.",
    "Use a contrast between what someone expected and what happened.",
)


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def digest_file(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def build_prompts(train_count: int, valid_count: int) -> list[dict]:
    import random
    rng = random.Random(20261106)
    train_pairs = [(style, topic) for style in STYLES for topic in TOPICS_TRAIN]
    valid_pairs = [(style, topic) for style in STYLES for topic in TOPICS_VALID]
    rng.shuffle(train_pairs)
    rng.shuffle(valid_pairs)
    train = [{"split": "train", "id": f"train-{i:04d}",
              "prompt": f"{style} Write about {topic}."}
             for i, (style, topic) in enumerate(train_pairs[:train_count])]
    valid = [{"split": "heldout", "id": f"heldout-{i:04d}",
              "prompt": f"{style} Write about {topic}."}
             for i, (style, topic) in enumerate(valid_pairs[:valid_count])]
    return train + valid


def generate_teacher_data(out: Path, train_count: int = 192,
                          valid_count: int = 32, batch_size: int = 4) -> dict:
    if not 32 <= train_count <= 512 or not 8 <= valid_count <= 64:
        raise ValueError("Teacher example counts outside bounded pilot limits")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(min(2, torch.get_num_threads()))
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION,
                                               trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_REPO, revision=QWEN_REVISION, torch_dtype=torch.float32,
        low_cpu_mem_usage=True, trust_remote_code=False,
    ).to("cpu").eval()
    prompts = build_prompts(train_count, valid_count)
    records: list[dict] = []
    for start in range(0, len(prompts), batch_size):
        group = prompts[start:start + batch_size]
        chats = [[{"role": "user", "content": item["prompt"]}] for item in group]
        rendered = [tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True, enable_thinking=False
        ) for chat in chats]
        encoded = tokenizer(rendered, return_tensors="pt", padding=True,
                            truncation=True, max_length=256, add_special_tokens=False)
        torch.manual_seed(20261106 + start)
        with torch.inference_mode():
            generated = model.generate(
                **encoded, max_new_tokens=96, do_sample=True, temperature=0.7,
                top_p=0.9, repetition_penalty=1.08,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_width = encoded["input_ids"].shape[1]
        for item, ids in zip(group, generated[:, input_width:]):
            answer = tokenizer.decode(ids, skip_special_tokens=True).strip()
            if len(answer) < 35:
                raise ValueError(f"Qwen returned an empty/short response for {item['id']}")
            if "<|" in answer:
                answer = answer.split("<|", 1)[0].strip()
            records.append({**item, "response": answer,
                            "teacher": QWEN_REPO, "teacher_revision": QWEN_REVISION})
        print(json.dumps({"generated": len(records), "requested": len(prompts)}), flush=True)

    normalized = [" ".join(row["response"].casefold().split()) for row in records]
    duplicate_responses = len(normalized) - len(set(normalized))
    if duplicate_responses > max(2, len(records) // 50):
        raise ValueError("Unexpectedly high exact duplicate rate in teacher responses")
    out.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records)
    data_path = out / "teacher-data.jsonl"
    data_path.write_text(data, encoding="utf-8")
    manifest = {
        "schema": 1, "method": "Qwen3 sequence-level response targets; inference only",
        "model": QWEN_REPO, "revision": QWEN_REVISION, "license": QWEN_LICENSE,
        "license_source": "https://huggingface.co/Qwen/Qwen3-0.6B",
        "generation": {"seed": 20261106, "temperature": 0.7, "top_p": 0.9,
                       "max_new_tokens": 96, "thinking": False},
        "train_count": train_count, "heldout_count": valid_count,
        "exact_duplicate_responses": duplicate_responses,
        "teacher_data_sha256": digest_file(data_path),
        "note": "Synthetic teacher outputs are experimental targets, not verified facts or user memory.",
    }
    (out / "teacher-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    del model
    return manifest


def teacher_stream(path: Path, tokenizer: Tokenizer, split: str) -> torch.Tensor:
    texts = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("split") == split:
            texts.append(f"User: {row['prompt']}\nAssistant: {row['response']}\n")
    if not texts:
        raise ValueError(f"No teacher examples for split {split}")
    ids: list[int] = []
    separator = tokenizer.encode("\n\n").ids
    for text in texts:
        ids.extend(tokenizer.encode(text).ids)
        ids.extend(separator)
    if len(ids) <= SEQ_LEN + 1:
        raise ValueError("Teacher stream is too short for training/evaluation")
    return torch.tensor(ids, dtype=torch.long)


def batch_for_lane(train_ids: torch.Tensor, teacher_ids: torch.Tensor,
                   lane: str, generator: torch.Generator):
    if lane == "uniform":
        return make_batch(train_ids, BATCH, SEQ_LEN, generator)
    xs, ys = [], []
    for _ in range(BATCH):
        use_teacher = bool(torch.rand((), generator=generator).item() < 0.10)
        data = teacher_ids if use_teacher else train_ids
        start = int(torch.randint(0, data.numel() - SEQ_LEN - 1,
                                  (1,), generator=generator).item())
        xs.append(data[start:start + SEQ_LEN])
        ys.append(data[start + 1:start + SEQ_LEN + 1])
    return torch.stack(xs), torch.stack(ys)


def run_training(source: Path, teacher_dir: Path, out: Path, lane: str,
                 steps: int = 8000, seed: int = 43) -> dict:
    if lane not in {"uniform", "qwen"} or not 1 <= steps <= 8000:
        raise ValueError("Expected lane=uniform|qwen and 1–8000 updates")
    parent_path = source / "checkpoint.pt"
    parent_sha = digest_file(parent_path)
    if parent_sha != PARENT_SHA256:
        raise ValueError(f"Wrong parent checkpoint SHA-256: {parent_sha}")
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    if parent.get("step") != PARENT_STEP or parent.get("seed") != 43:
        raise ValueError("Expected the verified 95k scratch-origin parent")
    if digest_file(source / "tokenizer.json") != TOKENIZER_SHA256:
        raise ValueError("Forge tokenizer SHA-256 mismatch")
    tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    _, train_ids, valid_ids, test_ids, manifest = load_language_data(source)
    teacher_path = teacher_dir / "teacher-data.jsonl"
    teacher_manifest = json.loads((teacher_dir / "teacher-manifest.json").read_text())
    if teacher_manifest.get("model") != QWEN_REPO or teacher_manifest.get("revision") != QWEN_REVISION:
        raise ValueError("Unexpected teacher model/revision")
    if digest_file(teacher_path) != teacher_manifest.get("teacher_data_sha256"):
        raise ValueError("Teacher data hash does not match its manifest")
    teacher_train = teacher_stream(teacher_path, tokenizer, "train")
    teacher_valid = teacher_stream(teacher_path, tokenizer, "heldout")

    out.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "source-manifest.json", "train.txt", "valid.txt"):
        shutil.copy2(source / name, out / name)
    (out / "test.txt").write_text(
        (source / "test.txt").read_text(encoding="utf-8") if (source / "test.txt").exists()
        else "", encoding="utf-8")
    (out / "evaluation-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(teacher_path, out / "teacher-data.jsonl")
    shutil.copy2(teacher_dir / "teacher-manifest.json", out / "teacher-manifest.json")

    torch.set_num_threads(min(2, torch.get_num_threads()))
    model = ByteTransformer(ModelConfig(**parent["configuration"]))
    model.load_state_dict(parent["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(parent["lr"]))
    optimizer.load_state_dict(parent["optimizer"])
    torch.set_rng_state(parent["torch_rng_state"])
    generator = torch.Generator()
    generator.set_state(parent["sample_rng_state"])
    initial_valid = heldout(model, valid_ids, windows=EVAL_WINDOWS, seq_len=256)
    initial_test = heldout(model, test_ids, windows=EVAL_WINDOWS, seq_len=256)
    initial_teacher = heldout(model, teacher_valid, windows=min(256, max(16, teacher_valid.numel() // 256)), seq_len=256)
    curve = []
    model.train()
    for update in range(1, steps + 1):
        x, y = batch_for_lane(train_ids, teacher_train, lane, generator)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update % 1000 == 0 or update == steps:
            valid_score = heldout(model, valid_ids, windows=EVAL_WINDOWS, seq_len=256)
            curve.append({"updates": update, "valid_loss_256_windows": round(valid_score, 8)})
            print(json.dumps({"lane": lane, "update": update,
                              "valid_loss_256_windows": round(valid_score, 8)}), flush=True)

    final_valid = heldout(model, valid_ids, windows=EVAL_WINDOWS, seq_len=256)
    final_test = heldout(model, test_ids, windows=EVAL_WINDOWS, seq_len=256)
    final_teacher = heldout(model, teacher_valid, windows=min(256, max(16, teacher_valid.numel() // 256)), seq_len=256)
    model.eval()
    ckpt = {
        **{k: v for k, v in parent.items() if k not in {"model", "optimizer", "torch_rng_state", "sample_rng_state"}},
        "step": PARENT_STEP + steps, "lane": lane, "lane_updates": steps,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(), "sample_rng_state": generator.get_state(),
        "teacher_model": QWEN_REPO, "teacher_revision": QWEN_REVISION,
        "teacher_data_sha256": teacher_manifest["teacher_data_sha256"],
        "distillation_fraction": 0.10 if lane == "qwen" else 0.0,
        "evaluation_manifest": manifest,
    }
    tmp = out / "checkpoint.pt.tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, out / "checkpoint.pt")
    report = {
        "schema": 1, "status": "research_only_not_promoted", "lane": lane,
        "teacher_model": QWEN_REPO, "teacher_revision": QWEN_REVISION,
        "teacher_license": QWEN_LICENSE, "teacher_data_sha256": teacher_manifest["teacher_data_sha256"],
        "parent_run": PARENT_RUN, "parent_step": PARENT_STEP,
        "parent_checkpoint_sha256": parent_sha, "candidate_step": PARENT_STEP + steps,
        "candidate_checkpoint_sha256": digest_file(out / "checkpoint.pt"),
        "updates": steps, "mix_fraction": 0.10 if lane == "qwen" else 0.0,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "train_sha256": manifest["train_sha256"], "valid_sha256": manifest["valid_sha256"],
        "independent_test_sha256": manifest["test_sha256"],
        "evaluation": {
            "method": "2,048 evenly spaced 256-token windows; same fixed tokenizer and splits",
            "parent_validation_loss": round(initial_valid, 8),
            "candidate_validation_loss": round(final_valid, 8),
            "parent_independent_test_loss": round(initial_test, 8),
            "candidate_independent_test_loss": round(final_test, 8),
            "parent_qwen_heldout_loss": round(initial_teacher, 8),
            "candidate_qwen_heldout_loss": round(final_teacher, 8),
            "curve": curve, "lower_is_better": True,
        },
        "limitations": ["Single matched seed; exploratory comparison, not repeatable-gain evidence.",
                        "Teacher outputs are synthetic language targets, not verified facts.",
                        "Text prediction and generated samples do not establish comprehension."],
        "android_integrated": False,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    (out / "lineage.json").write_text(json.dumps({
        "parent_run": PARENT_RUN, "parent_step": PARENT_STEP,
        "parent_sha256": parent_sha, "candidate_sha256": report["candidate_checkpoint_sha256"],
        "source_revision": manifest["revision"], "source_license": manifest["license"],
        "tokenizer_sha256": TOKENIZER_SHA256, "teacher": QWEN_REPO,
        "teacher_revision": QWEN_REVISION, "teacher_license": QWEN_LICENSE,
        "teacher_data_sha256": teacher_manifest["teacher_data_sha256"],
        "optimizer_state_resumed_from_parent": True,
        "forge_weights_initialized_from_own_verified_checkpoint": True,
        "promoted": False, "android_integrated": False,
    }, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--out", type=Path, required=True)
    gen.add_argument("--train-count", type=int, default=192)
    gen.add_argument("--valid-count", type=int, default=32)
    train = sub.add_parser("train")
    train.add_argument("--source", type=Path, required=True)
    train.add_argument("--teacher-dir", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--lane", choices=("uniform", "qwen"), required=True)
    train.add_argument("--steps", type=int, default=8000)
    args = parser.parse_args()
    if args.command == "generate":
        print(json.dumps(generate_teacher_data(args.out, args.train_count, args.valid_count), indent=2))
    else:
        run_training(args.source, args.teacher_dir, args.out, args.lane, args.steps)


if __name__ == "__main__":
    main()
