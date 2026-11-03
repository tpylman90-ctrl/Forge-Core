"""A calibration control: can Forge memorize sixteen repeated QA examples?

This is NOT a language-understanding experiment. A failure points to an
optimization/representation problem before harder generalization tasks.
Only original source checkpoint and original public synthetic QA fixtures.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import time

import torch

from forge_core.neural.model import ByteTransformer, ModelConfig
from forge_core.neural.preschool_qa import (
    training_tasks, encode_tasks, classify_batch, _score, verify_source,
)


def probe(folder: Path, out: Path, steps: int = 600, seed: int = 99,
          lr: float = 3e-4) -> dict:
    if not 1 <= steps <= 2000 or not 0 < lr < 0.01:
        raise ValueError("Invalid bounded memorization probe settings")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    checkpoint, manifest, tokenizer = verify_source(folder)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = ByteTransformer(ModelConfig(**checkpoint["configuration"]))
        model.load_state_dict(checkpoint["model"])
    tasks = training_tasks(16, seed)
    samples = encode_tasks(tasks, tokenizer, model.cfg.max_seq_len)
    a, b = tokenizer.encode("A").ids, tokenizer.encode("B").ids
    if len(a) != 1 or len(b) != 1 or a == b:
        raise ValueError("Choice labels cannot be scored as distinct single tokens")
    choices = (a[0], b[0])
    before = _score(model, tokenizer, tasks)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(seed + 1)
    model.train()
    curve = []
    start = time.monotonic()
    for step in range(1, steps+1):
        indices = [rng.randrange(len(samples)) for _ in range(4)]
        loss = classify_batch(model, [samples[i] for i in indices], choices)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            model.eval()
            score = _score(model, tokenizer, tasks)
            point = {"step": step,
                     "train_quiz_loss": round(float(loss.detach()), 5),
                     "same_example_accuracy": score["accuracy"]}
            curve.append(point)
            print(json.dumps({"calibration_progress":point}), flush=True)
            model.train()
    after = _score(model, tokenizer, tasks)
    result = {
        "schema": 1, "status": "memorization_sanity_only",
        "original_source_step": checkpoint["step"],
        "model_parameters": model.num_parameters,
        "source_manifest_train_sha256": manifest["train_sha256"],
        "seed": seed, "steps": steps, "examples": len(tasks),
        "baseline_training_accuracy": before,
        "final_same_training_accuracy": after, "curve": curve,
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "interpretation": (
            "If repeated fixed examples remain near chance, inspect the "
            "optimization and answer-token interface before enlarging data."
            " Memorization does not establish comprehension or transfer."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"initial": before, "after": after}, indent=2), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("artifacts/qa-calibration.json"))
    p.add_argument("--steps", type=int, default=600)
    a = p.parse_args()
    probe(a.checkpoint_dir, a.out, steps=a.steps)


if __name__ == "__main__":
    main()
