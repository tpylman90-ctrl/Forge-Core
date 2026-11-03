# Forge Core — Original Neural Language Training v0.3

## Canonical objective

Our long-term AI's original language network begins **randomly initialized**, not from inherited pretrained weights. The local pretrained teacher is an optional, architecturally separate language bridge that helps formulate questions while Forge Seed learns.

Two independent loops:

1. **Language:** approved text → quality filters/deduplication → byte tokens → next-byte loss → optimizer update → held-out evaluation → resumable checkpoint → repeat.
2. **Personal understanding:** opt-in observation → ask why → hypothesis → user confirmation → encrypted memory → application → evaluation. Confirmed memories are **not** training examples by default.

## What is implemented now

- Random-initialized decoder-only Transformer: original about 10M-parameter base and very small CPU smoke variant (actual count printed by each run).
- An intentionally simple **byte tokenizer**, 256 IDs, and standard next-token cross-entropy.
- Data loader for original tiny fixtures or **bounded** TinyStories dataset-viewer samples. Dataset and license recorded in a manifest.
- Separate *train* and *validation* files; normalized text; exact document deduplication; SHA256 checksums.
- A run compares its loss on fixed held-out windows before and after additional training steps, recording bits-per-byte and parameter count.
- Checkpoints include network weights, AdamW optimizer state, global step, RNG states, config, data hashes and metrics history.
- Resuming rejects a mismatched dataset, model configuration, learning rate, seed or batch settings.
- Manual GitHub Actions workflow with a checkpoint artifact and optional **prior run ID**. CPU runs are intentionally capped.

## Try a tiny original experiment locally

    python -m pip install -e .
    python -m forge_core.neural.prepare --source demo --out data/experiment
    python -m forge_core.neural.experiment --train data/experiment/train.txt --valid data/experiment/valid.txt --model-size tiny --steps 5 --out artifacts/checkpoint.pt --report artifacts/metrics.json

To resume the SAME experiment for 5 further optimizer steps:

    python -m forge_core.neural.experiment --train data/experiment/train.txt --valid data/experiment/valid.txt --model-size tiny --steps 5 --resume artifacts/checkpoint.pt --out artifacts/checkpoint.pt --report artifacts/metrics.json

The demo corpus is repetitive and **not** proof of language acquisition.

## Try bounded TinyStories

Dataset: https://huggingface.co/datasets/roneneldan/TinyStories

Current dataset-card license: **CDLA-Sharing-1.0**. Review its current terms before use and keep notices/provenance. Only download explicitly selected public samples, never private text.

    python -m forge_core.neural.prepare --source tinystories --out data/tinystories --train-rows 200 --valid-rows 50
    python -m forge_core.neural.experiment --train data/tinystories/train.txt --valid data/tinystories/valid.txt --model-size tiny --steps 20 --out artifacts/tinystories.pt

The bounded Hugging Face dataset viewer yields rows, **not necessarily complete stories**. This is a small API experiment, not the full dataset. For serious controlled training, ingest approved source documents with documented story boundaries, deduplicate near duplicates, create fixed held-out splits, and pin revisions/hashes. TinyStories-v1 and v2 data are not automatically interchangeable.

## GitHub Actions

Open **Actions → Forge Scratch Language Experiment → Run workflow** on the default branch (once merged).

Inputs: data source (demo or tinystories), model size (tiny/base), additional optimizer steps (1–30 tiny; 1–5 base), and optional prior successful workflow **run ID**.

The first run generates an artifact named *forge-scratch-language-checkpoint* containing:
- checkpoint.pt (weights + AdamW state + sampling RNG + training history)
- metrics.json (held-out results)
- dataset-manifest.json (source, license, splits, SHA256)

To continue the experiment, manually run the workflow with the same dataset/model/seed/config and specify the earlier run ID in *resume_run_id*. The workflow downloads that run's artifact and **fails closed** if the training corpus hash changes, preventing accidental continuation on different data.

**This is not yet an autonomous continuous training service.** There is no background schedule, automatic checkpoint promotion or permanent artifact store. GitHub-hosted runners provide modest CPU capacity; large models/data require separate authorized GPU compute. The public repository must never contain private memories, credentials, or private datasets.

GitHub artifacts have configurable and finite retention (public repository max 90 days). The current experiment retains artifacts for 30 days. Download and archive important checkpoints before they expire, using a separate approved store; GitHub Secrets is not checkpoint storage.

## Evaluation discipline

Every experiment must report:
- Source, license, exact source revision where possible, accepted bytes and hashes, train/validation separation
- Architecture, trainable parameters, tokenizer, context length and random seed
- Step count, batch size, learning rate, hardware and optimizer
- Held-out next-byte loss and bits-per-byte, initial vs final, comparisons across runs
- Simple readable samples and human evaluation in later versions
- Regressions and failure cases, not just improvements

If validation loss **increases**, checkpoint the result for diagnosis but do not promote it as better. A falling training loss alone is not evidence of generalization, conversational ability, independent judgment, or personal understanding.
