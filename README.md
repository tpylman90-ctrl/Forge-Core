# Forge Core — Original AI research

**Forge Core** is an independent, experimental AI project. It is not
SceneForge, AssetWright, or the existing Forge-ai Android application.

**Primary current objective: teach our own randomly initialized neural network
language and measure its progress.** We are not substituting a pretrained Qwen
model for the original Forge Core weights. Future identity, curiosity-driven
"ask why", user-controlled memory and Android interfaces follow evidence gates.

The five-year goal is a J.A.R.V.I.S.-like thinking partner with its own stable
personality that learns how and *why* its user makes decisions. It may propose
ideas independently but must obtain explicit authorization before external
actions. No current prototype meets that long-term vision.

## Development order

**L0 provenance → L1 real-text pretraining → L2 language and dialogue quality →
L3 independent identity → L4 personal understanding and memory → L5 Android.**

See [language-first roadmap](docs/ROADMAP.md) and the
[training pipeline](docs/LANGUAGE-TRAINING-LOOP.md).

## Verified versus experimental

- **Implemented:** A from-scratch causal Transformer, byte tokenizer,
  reproducible training/evaluation and resumable optimizer checkpoints.
- **Implemented:** Bounded TinyStories preview loader with document IDs,
  near-duplicate checks, manifest verification and fixed held-out windows.
- **Available sizes:** tiny ~25k smoke control; seed ~1.9M CPU experiment;
  base ~10.8M original architecture.
- **Verified previously:** Three steps on synthetic text in a 24,928-param
  model decreased sampled validation loss modestly. This is **not** language
  acquisition. See [experiment ledger](docs/EXPERIMENTS.md).
- **Unverified:** Meaningful natural-language ability from real data,
  conversation, reasoning, and personality.
- **Paused:** Expanding Android observer features. Existing Android source is
  preserved as an independent prototype, not the language foundation.

## Current bounded research run

Forge Seed Run 003 is a **one-time GitHub Actions continuation** of Run 002,
using the same 1,875,072-parameter randomly initialized network and
source-locked 200/50-document TinyStories preview. The new source content
lock at `data/locks/tinystories-preview-200-50-v0.4.json` validates the
**exact SHA256 bytes**, not a permanent upstream dataset revision.

The run adds 1,000 CPU optimization steps with validation every 100 steps,
a wider held-out audit and a fixed-seed text sample. Its GitHub artifact includes
the training checkpoint, optimizer and RNG state, exact public text slices,
dataset provenance and measured learning curve. The artifact expires after 30 days;
a separate reviewed archive is needed for long-term preservation.

It is important to distinguish statistical next-byte learning from coherent
sentences, instruction following and independent judgment. None is assumed.

## Run CPU tests

Python 3.10+ (plus PyTorch and cryptography):

    python -m pip install -e .
    python -m unittest discover -s tests -v

Run the tiny synthetic control:

    python -m forge_core.neural.prepare --source demo --out data/control
    python -m forge_core.neural.experiment --train data/control/train.txt \
      --valid data/control/valid.txt --manifest data/control/manifest.json \
      --model-size tiny --steps 5 --out artifacts/control.pt

For the first bounded real-language experiment, follow
[the training guide](docs/LANGUAGE-TRAINING-LOOP.md) or manually use the
"Forge Scratch Language Experiment" GitHub Actions workflow.

**Never commit** private memory, credentials, checkpoints, corpora or personal
observations to this public repository. Only properly licensed sources are
eligible for the language dataset. Generated checkpoints are not permanent
backups; audit the current data card license before scaling.

Founding governance:
[Charter](CHARTER.md) · [Architecture](docs/ARCHITECTURE.md) ·
[Evaluation](docs/EVALUATION.md) · [Privacy](SECURITY.md).

Our success measure is demonstrated language progress and useful behavior,
not training time, downloads, model size, or appearance of autonomy.

## License

Code license selection is pending. Do not presume rights to distribute this
code or third-party datasets beyond their documented terms.
