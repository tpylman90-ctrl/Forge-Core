"""Raw local chat with a research-only Forge Seed dialogue candidate.

This does not call a hosted model, add hidden reply templates, or access the
personal-memory vault. Replies are unfiltered candidate-model generations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.nn import functional as F

from forge_core.neural.model import ByteTransformer, ModelConfig


@torch.inference_mode()
def reply(model: ByteTransformer, codec: Tokenizer, turns: list[tuple[str, str]],
          message: str, *, max_new_tokens: int = 64, temperature: float = 0.8) -> str:
    history = turns[-4:] + [("User", message)]
    prompt = "Conversation:\n" + "\n".join(f"{role}: {text}" for role, text in history) + "\nForge:"
    ids = codec.encode(prompt).ids
    if len(ids) >= model.cfg.max_seq_len:
        raise ValueError("Conversation is too long for this candidate's context")
    x = torch.tensor([ids], dtype=torch.long)
    output: list[int] = []
    model.eval()
    for _ in range(max_new_tokens):
        logits, _ = model(x[:, -model.cfg.max_seq_len:])
        next_logits = logits[0, -1] / temperature
        # Keep sampling simple and reproducible: top-40 nucleus-like cutoff.
        values, indices = torch.topk(next_logits, k=min(40, next_logits.numel()))
        probs = F.softmax(values, dim=-1)
        token = int(indices[torch.multinomial(probs, 1)])
        output.append(token)
        x = torch.cat((x, torch.tensor([[token]], dtype=torch.long)), dim=1)
        decoded = codec.decode(output, skip_special_tokens=True)
        if "\n" in decoded:
            decoded = decoded.split("\n", 1)[0]
            break
    return codec.decode(output, skip_special_tokens=True).split("\n", 1)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True,
                        help="Research candidate checkpoint (for example correct.pt)")
    parser.add_argument("--tokenizer", type=Path, required=True,
                        help="Source Forge Seed tokenizer.json")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 128 or not 0.1 <= args.temperature <= 1.5:
        parser.error("Generation limits outside reviewed range")
    candidate = torch.load(args.candidate, map_location="cpu", weights_only=True)
    config = candidate.get("model_config")
    if not config or candidate.get("adapter_kind") != "full scratch-lineage candidate; research only":
        parser.error("Candidate lacks expected scratch-lineage research metadata")
    model = ByteTransformer(ModelConfig(**config))
    model.load_state_dict(candidate["model"], strict=True)
    codec = Tokenizer.from_file(str(args.tokenizer))
    if len(codec.get_vocab()) != model.cfg.vocab_size:
        parser.error("Tokenizer vocabulary does not match candidate model")
    print("Forge Seed research chat — raw model replies; no hosted model or memory.")
    print("This small experimental model may produce incoherent text. Type /exit to stop.\n")
    turns: list[tuple[str, str]] = []
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message.casefold() in {"/exit", "/quit"}:
            break
        if not message:
            continue
        answer = reply(model, codec, turns, message,
                       max_new_tokens=args.max_new_tokens,
                       temperature=args.temperature)
        print(f"Forge (raw): {answer}\n")
        turns.extend((("User", message), ("Forge", answer)))


if __name__ == "__main__":
    main()
