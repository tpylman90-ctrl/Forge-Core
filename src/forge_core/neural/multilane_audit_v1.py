"""Independent matched audit of Forge's 95k parent and 152k language candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.multilane_v1 import fetch_language_test
from forge_core.neural.seed_v07 import heldout


PARENT_SHA256 = "a7bcdb996e440f44d7496b887b54d19324927b4296fe9ed2f0f0293c88c63a0a"
TOKENIZER_SHA256 = "4ecd7803d4c369e80bda2e74c30cfb769b7e3a942f931df83383eec9f794f04c"
EVAL_WINDOWS = 2048
SEQ_LEN = 256

# Twenty newly written facts across four skills. Each is asked twice with the
# answer labels reversed to measure position bias.
QUESTIONS = [
    ("color", "Mara chose a cobalt kite. Jules chose an ochre drum.", "What color was Mara's kite?", "cobalt", "ochre"),
    ("color", "Tobin drew a silver moon. Rina drew a coral shell.", "What color was Rina's shell?", "coral", "silver"),
    ("color", "Eli packed a violet scarf. Noor packed a mint cup.", "What color was Eli's scarf?", "violet", "mint"),
    ("color", "Pia bought a bronze bell. Remy bought a lilac book.", "What color was Remy's book?", "lilac", "bronze"),
    ("color", "Asha carried an ivory flag. Luca carried a teal bucket.", "What color was the flag?", "ivory", "teal"),
    ("recipient", "Ivo passed the marble to Nia.", "Who received the marble?", "Nia", "Ivo"),
    ("recipient", "Keira handed the map to Soren.", "Who received the map?", "Soren", "Keira"),
    ("recipient", "Davi gave the blue cup to Esme.", "Who received the cup?", "Esme", "Davi"),
    ("recipient", "Yara sent the postcard to Felix.", "Who received the postcard?", "Felix", "Yara"),
    ("recipient", "Oren loaned the book to Zuri.", "Who received the book?", "Zuri", "Oren"),
    ("location", "The wooden button is inside the small jar.", "Where is the button?", "inside the jar", "beside the jar"),
    ("location", "A paper boat rests behind the flowerpot.", "Where is the boat?", "behind the flowerpot", "under the flowerpot"),
    ("location", "The brass key lies between two blue bowls.", "Where is the key?", "between the bowls", "above the bowls"),
    ("location", "A green mitten sits below the coat hook.", "Where is the mitten?", "below the hook", "near the hook"),
    ("location", "The toy drum is beside the red chair.", "Where is the drum?", "beside the chair", "inside the chair"),
    ("count", "Lena picked six berries. Amir picked nine berries.", "How many berries did Lena pick?", "six", "nine"),
    ("count", "Vik found three smooth stones. Bea found eight.", "How many stones did Vik find?", "three", "eight"),
    ("count", "Milo folded eleven paper stars. Ana folded four.", "How many stars did Milo fold?", "eleven", "four"),
    ("count", "Suri counted five yellow leaves. Teo counted two.", "How many leaves did Teo count?", "two", "five"),
    ("count", "Nell stacked seven blocks. Omar stacked ten.", "How many blocks did Nell stack?", "seven", "ten"),
]

GENERATION_PROMPTS = [
    "Write one clear sentence about a lantern beside a river.",
    "A child gives a blue scarf to a neighbor. Who receives the scarf?",
    "Explain in one short sentence why a kite moves when the wind blows.",
    "Complete this thought with one sentence: The cups are on the table because",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(folder: Path, expected_sha: str | None, expected_step: int):
    ckpt_path = folder / "checkpoint.pt"
    digest = sha256_file(ckpt_path)
    if expected_sha is not None and digest != expected_sha:
        raise ValueError(f"Checkpoint SHA mismatch for {folder}: {digest}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if ckpt.get("step") != expected_step:
        raise ValueError(f"Unexpected checkpoint step in {folder}: {ckpt.get('step')}")
    model = ByteTransformer(ModelConfig(**ckpt["configuration"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt, digest


def make_prompts() -> list[dict]:
    result = []
    for index, (skill, story, question, correct, distractor) in enumerate(QUESTIONS):
        for flip in (0, 1):
            left, right = ((correct, distractor) if flip == 0 else (distractor, correct))
            prompt = (
                f"Read the facts: {story}\nQuestion: {question}\n"
                f"A) {left}\nB) {right}\nAnswer:"
            )
            result.append({
                "id": f"fresh-{skill}-{index}-{flip}",
                "pair_id": f"fresh-{skill}-{index}",
                "skill": skill,
                "prompt": prompt,
                "gold": "A" if flip == 0 else "B",
            })
    return result


@torch.no_grad()
def mirrored_score(model: ByteTransformer, tokenizer: Tokenizer) -> dict:
    prompts = make_prompts()
    label_a, label_b = tokenizer.encode("A").ids, tokenizer.encode("B").ids
    if not label_a or not label_b or label_a[0] == label_b[0]:
        raise ValueError("A/B choices do not have distinct first tokens")
    pair_scores = {}
    by_skill = {}
    for task in prompts:
        prefix = tokenizer.encode(task["prompt"]).ids
        if len(prefix) >= model.cfg.max_seq_len:
            raise ValueError("Fresh capability prompt exceeds model context")
        logits, _ = model(torch.tensor([prefix], dtype=torch.long))
        log_probs = F.log_softmax(logits[0, -1, :], dim=0)
        answer = "A" if float(log_probs[label_a[0]]) >= float(log_probs[label_b[0]]) else "B"
        correct = int(answer == task["gold"])
        pair_scores.setdefault(task["pair_id"], []).append(correct)
        skill = by_skill.setdefault(task["skill"], {"correct": 0, "total": 0, "pairs_both_correct": 0})
        skill["correct"] += correct
        skill["total"] += 1
    for skill, data in by_skill.items():
        data["accuracy"] = round(data["correct"] / data["total"], 4)
        data["pairs_both_correct"] = sum(all(pair_scores[f"fresh-{skill}-{i}"]) for i in range(20)
                                         if f"fresh-{skill}-{i}" in pair_scores)
    fully_correct = sum(all(scores) for scores in pair_scores.values())
    return {
        "correct": sum(sum(scores) for scores in pair_scores.values()),
        "total": len(prompts),
        "accuracy": round(sum(sum(scores) for scores in pair_scores.values()) / len(prompts), 4),
        "mirrored_pairs": len(pair_scores),
        "both_orientations_correct": fully_correct,
        "one_orientation_correct": sum(sum(scores) == 1 for scores in pair_scores.values()),
        "both_orientations_wrong": sum(sum(scores) == 0 for scores in pair_scores.values()),
        "by_skill": by_skill,
        "prompt_set_sha256": hashlib.sha256(
            json.dumps(prompts, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "limitation": "Forced-choice diagnostic only; no claim of comprehension or reasoning.",
    }


@torch.no_grad()
def generate_ungraded(model: ByteTransformer, tokenizer: Tokenizer, seed: int) -> list[dict]:
    samples = []
    was_training = model.training
    model.eval()
    with torch.random.fork_rng():
        for index, prompt in enumerate(GENERATION_PROMPTS):
            torch.manual_seed(seed + index)
            ids = tokenizer.encode(prompt).ids
            output = model.generate(torch.tensor([ids], dtype=torch.long),
                                    new_tokens=64, temperature=0.7)
            samples.append({
                "prompt": prompt,
                "response": tokenizer.decode(
                    output[0].tolist()[len(ids):], skip_special_tokens=True
                ),
                "human_grade": "unreviewed",
            })
    model.train(was_training)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent-run", type=int, default=42143390701)
    parser.add_argument("--candidate-run", type=int, default=42149964829)
    parser.add_argument("--parent-step", type=int, default=95000)
    parser.add_argument("--candidate-step", type=int, default=152000)
    parser.add_argument("--parent-sha", default=PARENT_SHA256)
    parser.add_argument("--candidate-sha")
    args = parser.parse_args()
    torch.set_num_threads(min(2, torch.get_num_threads()))

    parent, parent_ckpt, parent_sha = load_model(args.parent, args.parent_sha, args.parent_step)
    candidate, candidate_ckpt, candidate_sha = load_model(args.candidate, args.candidate_sha, args.candidate_step)
    parent_tokenizer_path = args.parent / "tokenizer.json"
    candidate_tokenizer_path = args.candidate / "tokenizer.json"
    if sha256_file(parent_tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError("Parent tokenizer hash mismatch")
    if sha256_file(candidate_tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError("Candidate tokenizer hash mismatch")
    tokenizer = Tokenizer.from_file(str(parent_tokenizer_path))

    source_manifest = json.loads((args.parent / "source-manifest.json").read_text())
    for split in ("train", "valid"):
        if sha256_file(args.parent / f"{split}.txt") != source_manifest[f"{split}_sha256"]:
            raise ValueError(f"Parent {split} data hash mismatch")
    if candidate_ckpt.get("parent_sha256") != PARENT_SHA256:
        raise ValueError("Candidate lineage does not point to the verified 95k parent")

    test_text, audit_manifest = fetch_language_test(args.parent, source_manifest)
    test_tokens = torch.tensor(tokenizer.encode(test_text).ids, dtype=torch.long)
    losses = {}
    capabilities = {}
    generations = {}
    for name, model in (("parent_95k", parent), ("candidate_152k", candidate)):
        losses[name] = round(heldout(model, test_tokens, windows=EVAL_WINDOWS, seq_len=SEQ_LEN), 8)
        capabilities[name] = mirrored_score(model, tokenizer)
        generations[name] = generate_ungraded(model, tokenizer, 8300)

    task_bytes = json.dumps(make_prompts(), sort_keys=True).encode("utf-8")
    result = {
        "schema": 1,
        "status": "completed_research_audit_no_promotion",
        "parent": {
            "run": args.parent_run, "step": parent_ckpt["step"], "sha256": parent_sha,
        },
        "candidate": {
            "run": args.candidate_run, "step": candidate_ckpt["step"],
            "updates_from_parent": candidate_ckpt["step"] - parent_ckpt["step"],
            "sha256": candidate_sha,
        },
        "model_parameters": parent.num_parameters,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "independent_test": {
            "dataset": "Ronen Eldan TinyStories",
            "license": "CDLA-Sharing-1.0",
            "revision": source_manifest["revision"],
            "train_prefix_reserved_for_training_bytes": source_manifest["source_train_bytes_requested"],
            **audit_manifest["test_source"],
            "test_sha256": audit_manifest["test_sha256"],
            "documents": audit_manifest["test_document_count"],
            "utf8_bytes": audit_manifest["test_utf8_bytes"],
            "token_count": int(test_tokens.numel()),
            "window_count": EVAL_WINDOWS,
            "context_tokens": SEQ_LEN,
            "loss_by_checkpoint": losses,
            "candidate_minus_parent": round(
                losses["candidate_152k"] - losses["parent_95k"], 8
            ),
        },
        "fresh_mirrored_capability": {
            "task_count": 20,
            "orientation_count": 40,
            "task_sha256": hashlib.sha256(task_bytes).hexdigest(),
            "scores": capabilities,
        },
        "ungraded_generation_samples": generations,
        "claims": {
            "capability_claim": "None; this audit uses a small forced-choice diagnostic and ungraded text samples.",
            "promotion": "Neither candidate nor specialist branch is promoted by this audit.",
            "limitations": [
                "TinyStories is synthetic story text and does not measure general knowledge.",
                "The fresh A/B questions are template-based and remain a narrow diagnostic.",
                "Human review of generation samples was not performed.",
            ],
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "test.txt").write_text(test_text, encoding="utf-8")
    (args.out / "source-manifest.json").write_text(
        json.dumps(audit_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out / "report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": result["status"],
        "parent_loss": losses["parent_95k"],
        "candidate_loss": losses["candidate_152k"],
        "fresh_capability": {
            name: {k: v for k, v in score.items()
                   if k in ("correct", "total", "both_orientations_correct", "one_orientation_correct")}
            for name, score in capabilities.items()
        },
        "test_sha256": audit_manifest["test_sha256"],
        "report": str(args.out / "report.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
