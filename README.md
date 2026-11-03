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

## Master development direction

[**Forge Core Master Blueprint v1.0**](docs/MASTER-BLUEPRINT.md) governs
the research order. It distinguishes the **linguistic why** research loop
(verified language-structure comparisons, prototype now) from the later
**personal why** conversation/memory loop (explicit consent and confirmation).
The first [Linguistic Why Lab](docs/LINGUISTIC-WHY-LAB.md) ships only
deterministic gold-labeled probes and tests; it does not train or change model
weights. All additional learning objectives must beat controlled baselines
before adoption. The user remains final authority on charter changes.

## Development order

**L0 provenance → L1 real-text pretraining + linguistic probe design →
L2 tested linguistic supervision and coherent dialogue → L3 distinct identity →
L4 personal understanding/private memory → L5 user-approved actions and mobile.**

Designing the later layers can proceed early, but deploying them depends on
demonstrated language, privacy and authorization gates.

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

Forge Seed Run 003 **completed successfully as a one-time continuation** of Run 002,
using the same 1,875,072-parameter randomly initialized network and
source-locked 200/50-document TinyStories preview. The new source content
lock at `data/locks/tinystories-preview-200-50-v0.4.json` validates the
**exact SHA256 bytes**, not a permanent upstream dataset revision.

The run added 1,000 CPU optimization steps with validation every 100 steps,
a wider held-out audit and a fixed-seed text sample. Its GitHub artifact includes
the training checkpoint, optimizer and RNG state, exact public text slices,
dataset provenance and measured learning curve. The artifact expires after 30 days;
a separate reviewed archive is needed for long-term preservation.

The measured held-out loss fell from 3.581682 to 2.214508 in this continuation,
while the generated sample remained incoherent. See [the experiment ledger](docs/EXPERIMENTS.md).

It is important to distinguish statistical next-byte learning from coherent
sentences, instruction following and independent judgment. None is assumed.


### Linguistic Why research status

Three controlled auxiliary teaching runs have completed on copies of the
original 8,000-step Forge Seed. Public-language rehearsal prevented most
ordinary-language regression, but actor/patient understanding remained
**chance-level (160/320 held-out)** in the expanded third trial. These
experimental weights are **not** promoted. Read the
[experiment ledger](docs/EXPERIMENTS.md) and the
[linguistic why experiment report](docs/LINGUISTIC-WHY-EXPERIMENT.md)
before expanding training. The next gate is stronger language data,
tokenization and basic instruction-following benchmarks—not Android.

## Friday early-language milestone

Our next target is Forge Seed v0.7 by **Friday, November 6, 2026**.
This is a measurable language-development checkpoint, not a promise that
the model can already converse like a preschool child. The new run uses a
specific upstream TinyStories commit, a bounded larger public story sample,
an original BPE tokenizer learned from scratch and an original new neural
model initialized from random weights. Older checkpoints remain preserved.
See the [Friday Preschool Readiness Gate](docs/PRESCHOOL-FRIDAY.md).

## Run CPU tests

Python 3.10+ (plus PyTorch and cryptography):

    python -m pip install -e .
    python -m unittest discover -s tests -v

Run the tiny synthetic control:

    python -m forge_core.neural.prepare --source demo --out data/control
    python -m forge_core.neural.experiment --train data/control/train.txt \
      --valid data/control/valid.txt --manifest data/control/manifest.json \
      --model-size tiny --steps 5 --out artifacts/control.pt

For the expanded 800/200 public TinyStories experiment, see
[the experiment ledger](docs/EXPERIMENTS.md). Run 004 trained a fresh Forge
Seed for 4,000 steps, reducing fixed held-out next-byte loss from 5.5463 to
1.4101. Generated output is still not coherent English. Run 005 completed 8,000 total steps on the exact source-locked corpus,
reducing held-out loss further to 1.2533. The generated sample now has some
English-like clauses, but is not coherent enough for conversation.

Five fixed nonpersonal prompts have been compared through the
[language-quality inspection workflow](docs/LANGUAGE-QUALITY-TESTS.md).
Results show more familiar story syntax but still incoherent output. This is
**learning to predict language bytes, not a functional conversational AI**.
The 8,000-step research checkpoint has a validated, unpublished draft archive.

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

For long-lived, audited preservation of a public research checkpoint,
see the [draft archive procedure](docs/CHECKPOINT-ARCHIVE.md). Nothing
is automatically published as a trained conversational assistant.

### Latest measured language evidence (November 3)

The v0.7 original model at 14,000 steps now generates more readable story-like text (held-out next-token loss **2.6303**), but scored only **19/40** on balanced elementary A/B comprehension questions. A broad QA-training experiment failed to improve held-out accuracy (**20/40** verified vs **21/40** shuffled). A deliberately small sanity study then memorized all **16/16** examples after 100 updates, indicating our immediate weakness is *transferring* answer rules to unfamiliar sentences, not the ability to optimize an output token. A staged curriculum is running/being evaluated; experimental weights are never automatically promoted. See [Preschool Staged Research](docs/PRESCHOOL-STAGED.md) and the [Experiment Ledger](docs/EXPERIMENTS.md).
