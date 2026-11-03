# Forge Seed — Verified experiment ledger

## Run 001: from-scratch CPU pipeline validation

- **Date:** 2026-11-02 (US Eastern; UTC run date 2026-11-03)
- **GitHub workflow:** https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41549949410
- **Source:** original Forge Core generated demonstration sentences (NOT TinyStories)
- **Tokenizer:** byte-level (256 IDs); random-initialized Transformer
- **Model:** `tiny`, 24,928 trainable parameters (CI control, NOT Forge Seed's 10–30M target)
- **Training:** CPU, seed 42, 3 optimizer steps, batch 2, sequence length 32, AdamW learning rate 0.0003
- **Training:** 26,769 UTF-8 bytes; validation: 6,819 bytes
- **Held-out byte cross-entropy before:** 5.544311
- **Held-out byte cross-entropy after:** 5.493195
- **Relative difference:** approximately -0.92% (small improvement on fixed evaluation windows)
- **Held-out bits per byte after:** 7.925006
- **Artifact:** `forge-scratch-language-checkpoint`, GitHub artifact ID `13191740077`; retains checkpoint, AdamW optimizer state, RNG states, metrics history and dataset manifest; expires under repository retention policy.
- **Unit tests:** CI run https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41549948981 succeeded, including deterministic split/resume/mismatch tests.

### Interpretation

The training and evaluation code works under GitHub Actions, and the checkpoint artifacts were published. The model has not learned general language from this repetitive tiny fixture. This result is a reproducibility and bookkeeping milestone only.

### Next experiment

Run the bounded, license-reviewed `tinystories` source through **Actions → Forge Scratch Language Experiment → Run workflow**, then review held-out loss and save the checkpoint before its artifact expires. Scaling to 10–30M parameters and substantial corpus sizes requires separate compute planning and proper GPU access. No private personal memory belongs in these artifacts.


## Run 002: first real TinyStories preview experiment — Forge Seed v0.4

- **Date:** 2026-11-02 US Eastern (2026-11-03 UTC)
- **Workflow:** https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41554336734
- **Code SHA:** ec1307bf016551b62c8f80202b8efe8d91c1cefa
- **Source:** 200 public TinyStories training rows, 50 separate validation rows;
  dataset viewer preview, **not revision pinned** or a complete training corpus
- **Manifest:** 200 accepted training documents, 50 accepted validation
  documents, source/license, per-document IDs/hashes and split checks
- **Neural model:** original randomly initialized seed Transformer,
  **1,875,072** parameters, byte tokenizer, context 256
- **Optimization:** 20 AdamW steps, CPU, random seed 42, batch 2,
  training window 64 bytes, learning rate 0.0003
- **Data:** train 159,551 bytes; held-out validation 33,832 bytes
- **Held-out next-byte cross-entropy BEFORE:** **5.532837**
- **Held-out next-byte cross-entropy AFTER:** **3.581682**
- **Held-out bits per byte after:** **5.167275**
- **Comparison:** about **35.3%** relative held-out loss reduction over 16
  fixed, distributed evaluation windows in this one run
- **Generated sample:** contains many invalid/unreadable bytes; coherent
  language is **not** demonstrated
- **Checkpoint:** GitHub artifact forge-scratch-language-checkpoint,
  ID **13193865863**, includes optimizer, RNG state, manifest and metrics;
  expires **2026-12-03 UTC** if not copied to durable storage
- **CI:** https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41554336753
  passed the independent unit tests and smoke run

### Interpretation and limits

Forge can learn statistics of real held-out text using its own randomly
initialized weights. The small sample and 20-step run cannot establish
language proficiency, comprehension, conversation, or independent judgment.
The observed loss reduction may primarily reflect learning byte frequencies
and local patterns. The next experiment should use a longer controlled run,
a fixed/provenance-pinned corpus, more comprehensive held-out scoring, and
readable generation tests before promoting any capability claim.
