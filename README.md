# Forge Core

**Forge Core** is an independent, experimental personal AI research project. It is not the SceneForge game, AssetWright, or the existing Forge-ai Android application.

Our long-term ambition is a trustworthy, proactive thinking partner that learns *why* a person makes decisions and develops a useful, consistent perspective of its own. That is a research goal, **not** a claim of current capability or consciousness.

## Project rules

- Learn intentions, reasoning, outcomes and corrections rather than just memorizing actions.
- Think independently; **external actions by the future Forge agent require explicit, scoped user authorization**.
- Keep private user memories encrypted on-device by default, separate from neural-network weights.
- Treat public training datasets as untrusted inputs requiring licensing and provenance review.
- Make experiments reproducible and report measured improvements honestly.
- Separate identity, memory, learning and the neural foundation so components can evolve independently.

See [Founding Charter](CHARTER.md), [Architecture](docs/ARCHITECTURE.md), [Roadmap](docs/ROADMAP.md), [First Breath guide](docs/FIRST-BREATH.md), and [Security](SECURITY.md).

## v0.2 objectives

1. Implement a small causal Transformer initialized from random weights.
2. Train on locally supplied, properly licensed plain-text datasets; never on personal memory by default.
3. Validate data flow, causal masking, checkpointing and a CPU smoke test.
4. Prototype curiosity records and an authorization gate before connecting external tools.
5. Establish reproducible GitHub Actions tests.

## Quick start

Requires Python 3.10+.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python -m forge_core.cli demo
# Interactive teach/confirm/review/forget (local encrypted vault):
python -m forge_core.cli learn
python -m forge_core.cli review
```

Train only on a text file you have reviewed and are licensed to use, kept outside Git:

```bash
mkdir -p data
# Place your approved UTF-8 training text at data/train.txt
python -m forge_core.neural.train --text data/train.txt --steps 20 --batch-size 2 --seq-len 64
```

The default ~10M-parameter configuration is an **experimental** model, not a capable assistant. Significant training will need GPUs; the tiny CI configuration exists only to catch regressions. Model checkpoints, training corpora and private memory are ignored by Git.

## Status

The v0.2 research scaffold includes a functioning, template-based, user-driven learning CLI and a password-encrypted local memory prototype. A trained intelligent assistant, automatic phone observation, secure Android memory, and automatic model learning from experience are **not implemented**. CI workflow is present; verify its actual result before making success claims.

## License

Source-code license selection is pending; do not assume code or third-party datasets have an open license until one is explicitly added/reviewed.
