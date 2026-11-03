"""Optional language-teacher adapter for a locally installed llama.cpp binary.

The pretrained teacher can formulate questions now while Forge's original
scratch-initialized network trains separately. Nothing is downloaded or sent
to cloud services. The user must supply a local GGUF and llama-cli binary.

Model-generated questions are untrusted suggestions, not confirmed memories.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from typing import Callable, Sequence


class LocalTeacher:
    def __init__(
        self, binary: Path, model: Path, *, max_tokens: int = 128,
        timeout_seconds: int = 180,
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        if not binary.is_file() or not model.is_file():
            raise FileNotFoundError("Install llama.cpp locally and supply an existing GGUF model")
        if not (1 <= max_tokens <= 512) or timeout_seconds <= 0:
            raise ValueError("Invalid inference limits")
        self.binary = binary
        self.model = model
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.run = run

    def question(self, observation: str, confirmed_principles: Sequence[str] = ()) -> str:
        if not observation.strip():
            raise ValueError("Observation is required")
        # Do not interpolate a network address, shell or remote API.
        # Consent to sharing each observation with a local teacher still belongs
        # to the enclosing application; nothing runs automatically here.
        trusted = "\n".join("- " + p for p in confirmed_principles[:8]) or "(none)"
        prompt = (
            "You are the language teacher for the experimental Forge Core AI. "
            "Your role is to ask ONE curious, respectful, specific question about WHY "
            "a user made a choice. You must not assume their motivations, assert "
            "that you observed something not supplied, or invent principles. "
            "Do not ask for passwords, banking details, or other sensitive information. "
            "Do not perform actions. Reply only with the question.\n\n"
            "Confirmed principles (user approved):\n" + trusted +
            "\n\nUser-provided observation:\n" + observation.strip() +
            "\n\nQuestion:"
        )
        completed = self.run(
            [str(self.binary), "-m", str(self.model), "-c", "2048",
             "-n", str(self.max_tokens), "-p", prompt, "--temp", "0.7"],
            capture_output=True, text=True, check=True,
            timeout=self.timeout_seconds,
        )
        answer = completed.stdout.strip()
        if not answer:
            raise RuntimeError("The local model generated no question")
        return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Optional local GGUF language-teacher demo")
    parser.add_argument("--llama-cli", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--observation", help="Ask a curiosity question about a supplied observation")
    mode.add_argument("--development-scores", type=Path,
                      help="JSON with aggregate six-skill development scores only")
    parser.add_argument("--proposal-out", type=Path,
                        help="Optional path to write vetted curriculum proposal JSON")
    args = parser.parse_args()
    teacher = LocalTeacher(args.llama_cli, args.gguf)
    if args.observation is not None:
        if args.proposal_out:
            parser.error("--proposal-out requires --development-scores")
        print(teacher.question(args.observation))
    else:
        import json
        scores = json.loads(args.development_scores.read_text(encoding="utf-8"))
        proposal = teacher.curriculum_proposal(scores)
        result = json.dumps(proposal, indent=2) + "\n"
        if args.proposal_out:
            args.proposal_out.parent.mkdir(parents=True, exist_ok=True)
            args.proposal_out.write_text(result, encoding="utf-8")
        print(result, end="")


if __name__ == "__main__":
    main()
