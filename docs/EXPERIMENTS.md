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
