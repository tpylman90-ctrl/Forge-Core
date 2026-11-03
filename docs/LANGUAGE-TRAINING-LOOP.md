# Forge Core — Original language learning pipeline v0.4

## Primary mission

We are building Forge's **own neural language foundation**, starting with
randomly initialized weights. This—not Android app-switch observation—is the
current engineering priority. The pretrained teacher remains an **optional
separate tool**, never the source of Forge Seed's parameters.

## Sequence of operations (do not skip gates)

1. **Provenance:** choose properly licensed public language text. Record the
   dataset identity, license, selected split, document IDs and file hashes.
2. **Preparation:** normalize text, reject malformed/exact duplicate documents,
   screen 5-word-shingle near duplicates, and keep training/validation documents
   independently identified. Near-duplicate screening is a preliminary safeguard
   and does not replace a professional contamination audit.
3. **Train from random initialization:** use byte-level next-token prediction,
   causal masking, backpropagation and AdamW.
4. **Evaluate on withheld documents:** compare same spread-out fixed validation
   windows before and after each step batch. Report loss and bits per byte.
5. **Inspect generated samples:** optional fixed-seed "Once upon a time" sample.
   Small models may output nonsense; never conflate numeric loss improvement with
   useful natural language.
6. **Checkpoint and resume:** save weights, optimizer state, RNG, data hashes,
   manifest SHA, model config and previous metrics. Reject incompatible resumes.
7. **Repeat and scale only when measured progress justifies it.**

## Model ladder

| Mode | Approximate parameters | Purpose |
| --- | ---: | --- |
| tiny | 24,928 | Unit testing and workflow controls, not language |
| seed | ~1.9M | First CPU-feasible scratch-language training |
| base | ~10.8M | Original starting architecture for future GPU-scale work |

The 10–30M vision remains. The intermediate seed avoids spending expensive
CPU time on a ~10M model before validating a genuine learning trajectory.

## What is implemented, not what is proven

- Bounded public TinyStories dataset-viewer rows (or project-owned demo text).
  For now, **this is not a revision-pinned full corpus**. Data hashes and
  split-specific document IDs make individual runs auditable, but external
  dataset snapshots may change and cannot automatically resume.
- A checked schema-v2 manifest with source/license, accepted/rejected counts
  and train/validation document hashes.
- Fixed coverage of 16 held-out windows per default experiment, compared before
  and after training. This is more representative than two random windows, but
  still only a sampled evaluation.
- A reproducible optional text sample and checkpoint containing training state.
- A limited CPU GitHub Actions workflow, not an always-on autonomous trainer.
- We do **not** yet have evidence that Forge can understand natural language.

## Try a reproducible local experiment

    python -m pip install -e .
    python -m forge_core.neural.prepare --source demo --out data/test
    python -m forge_core.neural.experiment \
      --train data/test/train.txt --valid data/test/valid.txt \
      --manifest data/test/manifest.json \
      --model-size tiny --steps 5 --eval-batches 8 \
      --out artifacts/first.pt --report artifacts/first.json

For a **real public-text preview** (source license must be reviewed before use):

    python -m forge_core.neural.prepare --source tinystories \
      --out data/tinystories --train-rows 200 --valid-rows 50
    python -m forge_core.neural.experiment \
      --train data/tinystories/train.txt --valid data/tinystories/valid.txt \
      --manifest data/tinystories/manifest.json \
      --model-size seed --steps 20 --batch-size 2 --seq-len 64 \
      --eval-batches 8 --sample-tokens 48 \
      --out artifacts/seed.pt --report artifacts/seed.json

Source: https://huggingface.co/datasets/roneneldan/TinyStories
Dataset card currently labels it CDLA-Sharing-1.0. Confirm current terms,
provenance, and text usage before any larger training.

## Run 003: verified extended continuation (completed 1,020 total steps)

The workflow's one-time request at
`experiments/requests/seed-run003.md` resumes **Run 002**
(`41554336734`) at 20 optimizer steps, provided the upstream public
TinyStories preview still produces the **identical manifest SHA256** recorded
in `data/locks/tinystories-preview-200-50-v0.4.json`.

This is a **content lock** rather than a pinned immutable upstream revision.
If the remote data changes, training fails rather than silently continuing
on different text. Future full-corpus training still needs immutable source
revisions and an appropriate license/contamination audit.

Run 003 completed 1,000 additional CPU steps, keeping the exact prior network
weights/optimizer/RNG, and records validation loss every 100 steps. A second
held-out audit covers 128 fixed windows. The artifact now carries the exact
public train/validation slices and a notice to make the experiment resumable
within the limited retention period; these data must never include personal
memories.

The held-out loss **did improve** (3.581682 to 2.214508 on 16 windows; 3.582403 to 2.127761 on 128 windows). The generated text remains largely incoherent. **Text readability must be checked separately**.
We will not promote this to a functioning conversational AI without evidence.

## GitHub Actions

**Actions → Forge Scratch Language Experiment → Run workflow.**
Manual runs default to *tinystories + seed + 1,000 steps*, and must explicitly specify a prior run ID to resume. The one-time Run 003 push trigger is limited to its request marker and is not a recurring schedule.

For continuation, specify a previous run ID with the **same corpus and
settings**. Artifacts include checkpoint.pt, metrics.json and dataset-manifest.json.
They expire after 30 days. A mismatched data hash or manifest refuses a resume.
Never treat temporary GitHub artifacts as permanent model storage. Keep personal
observations, private memories, passwords and keys out of training and CI.

Standard hosted GitHub runners have CPU, not dedicated training GPUs; access
is subject to platform rules. No claim of a five-year autonomous learning agent
follows from these small experiments.

## Second learning loop — later

Observe (explicit opt-in) → Ask Why → Understand → Propose Principle →
**Confirm** → Remember privately → Apply → Evaluate.

That is a separate personal learning pipeline, not immediate weight updates.
It will be connected only after basic language quality is demonstrated and
private-memory/security requirements are tested.

## Preserving a successful research experiment

See [checkpoint archive](CHECKPOINT-ARCHIVE.md). The public-data-only
archive workflow audits the checkpoint, exact corpus and manifest before
creating a **DRAFT** GitHub Release. Neither private memory nor phone
observations may enter the public archive. Draft archive existence is not
proof of language quality or a guarantee of permanent storage.
