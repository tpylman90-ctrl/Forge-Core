"""Independent, fixed-prompt language inspection of a scratch Forge checkpoint.

This is a transparent qualitative benchmark, *not* an automated fluency score.
No personal data, pretrained weights, network calls, or external actions.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from forge_core.neural.model import ByteTransformer, ModelConfig

PROMPTS = (
    "Once upon a time",
    "The little girl",
    "A small dog",
    "One day the boy",
    "The old man said",
)


@torch.no_grad()
def inspect(checkpoint_path: Path, *, generated_bytes: int = 96,
            seed: int = 780, temperature: float = 0.7) -> dict:
    if not 1 <= generated_bytes <= 256 or not 0 < temperature <= 2:
        raise ValueError("Generation limits invalid")
    # Use only a checkpoint from a trusted Forge training workflow.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = ModelConfig(**ckpt["config"])
    results = []
    # Preserve the caller's RNG during both initialization and generation.
    with torch.random.fork_rng():
        model = ByteTransformer(cfg)
        model.load_state_dict(ckpt["model"])
        model.eval()
        for i, prompt in enumerate(PROMPTS):
            prefix = list(prompt.encode("utf-8"))
            if len(prefix) >= cfg.max_seq_len:
                raise ValueError("Prompt exceeds model context")
            torch.manual_seed(seed + i)
            tokens = model.generate(
                torch.tensor([prefix], dtype=torch.long),
                new_tokens=generated_bytes, temperature=temperature
            )[0].tolist()
            continuation = bytes(tokens[len(prefix):]).decode("utf-8", errors="replace")
            results.append({
                "prompt": prompt,
                "continuation": continuation,
                "replacement_char_count": continuation.count("\ufffd"),
                "printable_fraction": round(
                    sum(c.isprintable() for c in continuation) /
                    max(1, len(continuation)), 4),
            })
    return {
        "schema": 1,
        "checkpoint_step": ckpt["step"],
        "parameters": model.num_parameters,
        "model_config": asdict(cfg),
        "seed": seed,
        "temperature": temperature,
        "generated_bytes_each": generated_bytes,
        "samples": results,
        "warning": "Readable letters or decreasing byte loss do not prove fluent language, comprehension or reasoning. Human review is required.",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--generated-bytes", type=int, default=96)
    p.add_argument("--seed", type=int, default=780)
    p.add_argument("--temperature", type=float, default=0.7)
    a = p.parse_args()
    report = inspect(a.checkpoint, generated_bytes=a.generated_bytes,
                     seed=a.seed, temperature=a.temperature)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
