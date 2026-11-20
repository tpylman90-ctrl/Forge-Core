"""Purity-preserving dialogue-skill pilot for Forge Seed.

All dialogue targets are produced by reviewed deterministic templates from
fictional slots. No teacher model, personal conversation, or pretrained
weights are used. The immutable scratch-language parent is adapted into
research-only candidates and compared with a shuffled-target control.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.seed_v07 import heldout

PARENT_RUN = 42552572467
PARENT_STEP = 107250
PARENT_SEED = 43
PARENT_SHA256 = "848553a81549c1269c360c5d07c57adde92a7816a0bc19b61ecc8028c93fee99"
TOKENIZER_SHA256 = "4ecd7803d4c369e80bda2e74c30cfb769b7e3a942f931df83383eec9f794f04c"
MAX_CONTEXT = 256
INTENTS = ("recall", "correction", "clarify", "uncertainty")

SLOTS = {
    "train": {
        "names": ("Mira", "Dax", "Lena", "Omar", "Tess", "Ravi", "Nia", "Cole"),
        "items": ("drone", "racer", "engine", "map", "helmet", "robot", "bridge", "track"),
        "values": ("blue", "silver", "quiet", "fast", "small", "heavy", "open", "rough"),
        "projects": ("Blender", "Godot", "garage", "track", "model", "scene", "vehicle", "course"),
    },
    "development": {
        "names": ("Iris", "Pax", "Suri", "Bo"),
        "items": ("cabin", "pod", "camera", "hangar"),
        "values": ("green", "bronze", "smooth", "wide"),
        "projects": ("shader", "workshop", "vehicle", "terrain"),
    },
    "exam": {
        "names": ("Vera", "Quin", "Zuri", "Eli"),
        "items": ("thruster", "cockpit", "vehicle", "platform"),
        "values": ("orange", "black", "stable", "low"),
        "projects": ("hangar", "racer", "interior", "world"),
    },
}

# Separate prompt families by split so the final exam tests behavior rather
# than memorized prompt prefixes.
TEMPLATES = {
    "train": {
        "recall": (
            "Chat so far:\nUser: The {item} is {value}.\nForge: I have that for this chat.\nUser: What is the {item}?\nForge:",
            "Chat so far:\nUser: My {item} feels {value}.\nForge: Okay.\nUser: How does it feel?\nForge:",
        ),
        "correction": (
            "Chat so far:\nUser: I am working in {old}.\nForge: Understood.\nUser: Correction, I am using {new}.\nForge:",
            "Chat so far:\nUser: The {item} is {old}.\nForge: Got it.\nUser: Actually, make that {new}.\nForge:",
        ),
        "clarify": (
            "Chat so far:\nUser: I have a {item} in this {project}. Can you improve it?\nForge:",
            "Chat so far:\nUser: This {project} needs work around the {item}. Make it better.\nForge:",
        ),
        "uncertainty": (
            "Chat so far:\nUser: I have a {item} in the {project}.\nForge: Okay.\nUser: What color is it?\nForge:",
            "Chat so far:\nUser: I am building a {project} around the {item}.\nForge: Okay.\nUser: How heavy is it?\nForge:",
        ),
    },
    "development": {
        "recall": ("Earlier you said the {item} was {value}. Remind me: what was it like?\nForge:",
                   "What description did you give for the {item}: {value} or something else?\nForge:"),
        "correction": ("I said I used {old}, but I meant {new}. Which one should you use now?\nForge:",),
        "clarify": ("The {item} inside the {project} needs work. Can you fix it?\nForge:",
                    "I want to revise the {project} part involving the {item}. Can you help?\nForge:"),
        "uncertainty": ("I mentioned a {item} for the {project}. Did I tell you its color?\nForge:",
                        "You know my {project} and {item}. What is its weight?\nForge:"),
    },
    "exam": {
        "recall": ("Conversation: The {item} is {value}. Later: tell me its description.\nForge:",
                   "You described the {item} as {value}. What did you say about it?\nForge:"),
        "correction": ("Conversation: My tool is {old}. Update: I switched to {new}. What is current?\nForge:",),
        "clarify": ("Could you revise the {project} around this {item}?\nForge:",
                    "What should I change about the {item} within the {project}?\nForge:"),
        "uncertainty": ("Conversation: I own a {item} in the {project}. What color did I say it was?\nForge:",
                        "I have a {project} with a {item}. Did I give you its weight?\nForge:"),
    },
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_rows(split: str, count_per_intent: int = 32, seed: int = 2049) -> list[dict]:
    if split not in SLOTS or not 8 <= count_per_intent <= 128:
        raise ValueError("Invalid split or bounded example count")
    rng = random.Random(seed + {"train": 3, "development": 303, "exam": 3003}[split])
    slots = SLOTS[split]
    rows = []
    for intent in INTENTS:
        if intent == "recall":
            combinations = [(item, value) for item in slots["items"] for value in slots["values"]]
            rng.shuffle(combinations)
        elif intent in ("clarify", "uncertainty"):
            combinations = [(item, project) for item in slots["items"] for project in slots["projects"]]
            rng.shuffle(combinations)
        else:
            pool = slots["projects"] + slots["values"]
            combinations = [(old, new) for old in pool for new in pool if old != new]
            rng.shuffle(combinations)
        unique_combinations_needed = max(
            1, (count_per_intent + len(TEMPLATES[split][intent]) - 1)
            // len(TEMPLATES[split][intent]))
        combinations = combinations[:unique_combinations_needed]
        for i in range(count_per_intent):
            template = TEMPLATES[split][intent][(i // len(combinations)) % len(TEMPLATES[split][intent])]
            if intent == "recall":
                item, value = combinations[i % len(combinations)]
                prompt = template.format(item=item, value=value)
                wrong_value = next(v for v in slots["values"] if v != value)
                target, distractor = f"The {item} is {value}.", f"The {item} is {wrong_value}."
            elif intent == "correction":
                old, new = combinations[i % len(combinations)]
                item = rng.choice(slots["items"])
                prompt = template.format(old=old, new=new, item=item)
                target, distractor = f"I will use {new} now.", f"I will keep using {old}."
            elif intent == "clarify":
                item, project = combinations[i % len(combinations)]
                prompt = template.format(item=item, project=project)
                if "{item}" in template:
                    target = f"What change do you want for the {item} in the {project}?"
                else:
                    target = f"What change do you want for the {item} in the {project}?"
                distractor = "I will change it now without asking."
            else:
                item, project = combinations[i % len(combinations)]
                prompt = template.format(item=item, project=project)
                target = ("You have not told me its color yet." if "color" in prompt
                          else "I do not know its weight from this chat.")
                unsupported = next(v for v in slots["values"] if v not in prompt)
                distractor = f"You said it is {unsupported}."
            rows.append({"id": f"{split}:{intent}:{i:03d}", "split": split,
                         "intent": intent, "prompt": prompt, "target": target,
                         "distractor": distractor})
    if len({r["prompt"] for r in rows}) != len(rows):
        raise ValueError("Duplicate generated prompts")
    return rows


def shuffled_targets(rows: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    output = [dict(row) for row in rows]
    for intent in INTENTS:
        indices = [i for i, row in enumerate(output) if row["intent"] == intent]
        values = [output[i]["target"] for i in indices]
        original = values[:]
        for _ in range(100):
            rng.shuffle(values)
            if values != original:
                break
        if values == original:
            raise ValueError("Unable to build shuffled-target control")
        for i, value in zip(indices, values):
            output[i]["target"] = value
    return output


def encode(rows: list[dict], codec: Tokenizer, context: int) -> list[tuple[list[int], list[int], str]]:
    examples = []
    for row in rows:
        prompt = codec.encode(row["prompt"]).ids
        answer = codec.encode(row["target"]).ids
        seq = prompt + answer
        if not prompt or not answer or len(seq) > context:
            raise ValueError(f"Invalid sequence length for {row['id']}")
        examples.append((seq, [len(prompt) - 1, len(seq) - 1], row["intent"]))
    return examples


def batch_loss(model: ByteTransformer, batch: list[tuple[list[int], list[int], str]]) -> torch.Tensor:
    width = max(len(seq) - 1 for seq, _, _ in batch)
    x = torch.zeros((len(batch), width), dtype=torch.long)
    y = torch.full((len(batch), width), -100, dtype=torch.long)
    for i, (seq, (start, end), _) in enumerate(batch):
        x[i, :len(seq)-1] = torch.tensor(seq[:-1])
        y[i, start:end] = torch.tensor(seq[start+1:end+1])
    logits, _ = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)


@torch.no_grad()
def score_candidates(model: ByteTransformer, rows: list[dict], codec: Tokenizer,
                     context: int) -> dict:
    """Choose between correct and intent-matched wrong responses by token NLL."""
    model.eval()
    by_intent: dict[str, list[dict]] = {k: [] for k in INTENTS}
    for row in rows:
        by_intent[row["intent"]].append(row)
    result = {k: {"correct": 0, "total": 0} for k in INTENTS}
    for intent, group in by_intent.items():
        for i, row in enumerate(group):
            prompt_ids = codec.encode(row["prompt"]).ids
            scores = []
            for candidate in (row["target"], row["distractor"]):
                ids = prompt_ids + codec.encode(candidate).ids
                if len(ids) > context:
                    raise ValueError("Evaluation candidate exceeds context")
                x = torch.tensor([ids[:-1]], dtype=torch.long)
                logits, _ = model(x)
                logp = F.log_softmax(logits[0], dim=-1)
                start = len(prompt_ids) - 1
                target_ids = ids[len(prompt_ids):]
                token_idx = torch.arange(start, start + len(target_ids))
                target_tensor = torch.tensor(target_ids)
                scores.append(float(-logp[token_idx, target_tensor].mean()))
            result[intent]["correct"] += int(scores[0] < scores[1])
            result[intent]["total"] += 1
    for intent in INTENTS:
        n = result[intent]["total"]
        result[intent]["accuracy"] = round(result[intent]["correct"] / max(1, n), 4)
    result["macro_accuracy"] = round(sum(result[k]["accuracy"] for k in INTENTS) / len(INTENTS), 4)
    return result


def run(*, source: Path, out: Path, steps: int = 160, seed: int = 43) -> dict:
    parent_path = source / "checkpoint.pt"
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    parent_sha = digest(parent_path)
    if (parent_sha != PARENT_SHA256 or int(parent.get("step", -1)) != PARENT_STEP
            or int(parent.get("seed", -1)) != PARENT_SEED):
        raise ValueError("Expected the authenticated Forge Seed v0.7 source checkpoint")
    codec_path = source / "tokenizer.json"
    codec_hash = digest(codec_path)
    codec = Tokenizer.from_file(str(codec_path))
    if codec_hash != TOKENIZER_SHA256 or codec_hash != parent.get("tokenizer_sha256"):
        raise ValueError("Tokenizer hash differs from source checkpoint")
    manifest = json.loads((source / "source-manifest.json").read_text())
    for split in ("train", "valid"):
        if digest(source / f"{split}.txt") != manifest.get(f"{split}_sha256"):
            raise ValueError(f"Pinned {split} text hash mismatch")
    train_ids = torch.tensor(codec.encode((source / "train.txt").read_text(encoding="utf-8")).ids,
                             dtype=torch.long)
    valid_ids = torch.tensor(codec.encode((source / "valid.txt").read_text(encoding="utf-8")).ids,
                             dtype=torch.long)
    if manifest.get("revision") != "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64" or manifest.get("license") != "cdla-sharing-1.0":
        raise ValueError("Language source is not the approved pinned TinyStories data")
    cfg = ModelConfig(**parent["configuration"])
    model_config_hash = hashlib.sha256(json.dumps(parent["configuration"], sort_keys=True).encode()).hexdigest()
    rows = {split: build_rows(split) for split in ("train", "development", "exam")}
    encoded = {k: encode(v, codec, cfg.max_seq_len) for k, v in rows.items()}
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(codec_path, out / "tokenizer.json")
    # Save evidence of lineage and synthetic data only. No private text is read.
    data_manifest = {"schema": 1, "parent_run": PARENT_RUN,
                     "parent_sha256": parent_sha, "parent_step": parent["step"],
                     "parent_seed": parent["seed"], "tokenizer_sha256": codec_hash,
                     "model_config_sha256": model_config_hash,
                     "source_revision": manifest["revision"], "source_license": manifest["license"],
                     "split_sha256": {k: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()
                                      for k, v in rows.items()},
                     "counts": {k: len(v) for k, v in rows.items()},
                     "targets": "deterministic reviewed templates; no teacher model or personal chat"}
    (out / "manifest.json").write_text(json.dumps(data_manifest, indent=2) + "\n")
    original = ByteTransformer(cfg)
    original.load_state_dict(parent["model"], strict=True)
    baseline = score_candidates(original, rows["exam"], codec, cfg.max_seq_len)
    tiny_before = heldout(original, valid_ids, windows=256, seq_len=128)
    outcomes = {}
    for arm, training_rows in (("correct", rows["train"]),
                               ("shuffled_control", shuffled_targets(rows["train"], seed + 1))):
        torch.manual_seed(seed)
        branch = ByteTransformer(cfg)
        branch.load_state_dict(parent["model"], strict=True)
        batch_rng = random.Random(seed + (0 if arm == "correct" else 1))
        data = encode(training_rows, codec, cfg.max_seq_len)
        opt = torch.optim.AdamW(branch.parameters(), lr=1e-5)
        curve = []
        branch.train()
        started = time.monotonic()
        for step in range(1, steps + 1):
            batch = [data[batch_rng.randrange(len(data))] for _ in range(4)]
            loss = batch_loss(branch, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(branch.parameters(), 1.0)
            opt.step()
            if step % max(1, steps // 4) == 0 or step == steps:
                curve.append({"step": step, "train_loss": round(float(loss.detach()), 6)})
        score = score_candidates(branch, rows["exam"], codec, cfg.max_seq_len)
        retention = heldout(branch, valid_ids, windows=256, seq_len=128)
        torch.save({"parent_run": PARENT_RUN, "parent_sha256": data_manifest["parent_sha256"],
                    "parent_step": parent["step"], "parent_seed": parent["seed"],
                    "adapter_kind": "full scratch-lineage candidate; research only",
                    "intent": arm, "model_config": parent["configuration"],
                    "model": branch.state_dict(), "updates": steps, "seed": seed},
                   out / f"{arm}.pt")
        outcomes[arm] = {"exam": score, "tiny_stories_validation_loss": round(retention, 6),
                         "training_curve": curve, "wall_seconds": round(time.monotonic()-started, 2)}
    correct_score = outcomes["correct"]["exam"]["macro_accuracy"]
    control_score = outcomes["shuffled_control"]["exam"]["macro_accuracy"]
    parent_score = baseline["macro_accuracy"]
    retention_delta = outcomes["correct"]["tiny_stories_validation_loss"] - round(tiny_before, 6)
    dialogue_gate = {
        "correct_arm_macro_accuracy": correct_score,
        "shuffled_control_macro_accuracy": control_score,
        "frozen_parent_macro_accuracy": parent_score,
        "correct_arm_beats_frozen_parent": correct_score > parent_score,
        "correct_arm_beats_shuffled_control": correct_score > control_score,
        "tiny_stories_validation_loss_delta": round(retention_delta, 6),
        "language_retention_limit": 0.05,
        "language_retention_passed": retention_delta <= 0.05,
        "passed": (correct_score > parent_score and correct_score > control_score
                   and retention_delta <= 0.05),
        "interpretation": "A pass is evidence only for these synthetic response-ranking tasks; it does not certify open-ended chat.",
    }
    report = {"status": "research_only_not_conversation_certified", "parent_run": PARENT_RUN,
              "parent_sha256": data_manifest["parent_sha256"], "parent_step": parent["step"],
              "parent_seed": parent["seed"], "steps_per_arm": steps,
              "tiny_stories_validation_loss_before": round(tiny_before, 6),
              "baseline_exam": baseline, "arms": outcomes,
              "dialogue_improvement_gate": dialogue_gate,
              "note": "Candidate ranking on synthetic dialogs is a narrow skill probe, not open-ended chat competence. No weights are promoted."}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True, help="Authenticated original Seed v0.7 checkpoint directory")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=160)
    p.add_argument("--seed", type=int, default=43)
    a = p.parse_args()
    if not 20 <= a.steps <= 500:
        p.error("steps must be between 20 and 500")
    run(source=a.source, out=a.out, steps=a.steps, seed=a.seed)


if __name__ == "__main__":
    main()
