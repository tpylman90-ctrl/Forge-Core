# Forge Seed v0.7 — Preschool readiness research run 001

Deadline goal: Friday, 2026-11-06; preschool is an evaluation target,
NOT a guaranteed achievement or a label applied automatically.

Authorized one-time public-data research experiment with GitHub Actions.

- Immutable source revision: roneneldan/TinyStories
  f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
  (dataset card marks CDLA-Sharing-1.0; keep attribution and review terms).
- Stream only first 10 MiB of TinyStories-train.txt and first 2 MiB
  of the upstream separate TinyStories-valid.txt. Keep complete stories and
  exact-hash deduplicate; don't download full 1.92GB corpus.
- Train 2048-vocabulary byte-level BPE from scratch on train only.
- Instantiate our original 4-layer, 4-head, width-192 decoder Transformer
  from random weights and the newly learned vocabulary, not previous Qwen
  or any pretrained model weights. Keep v0.6 checkpoint unchanged.
- Bounded 14,000 CPU optimization steps; batch 4, 128 BPE tokens context.
- Held-out loss every 2,000 steps, 256-window audit, six sampled stories.
- Save optimizer state, RNG, tokenizer, exact public sample,
  source manifest and metrics in 30-day research artifact.
- No personal data, Android observations, user memories, agent actions,
  or third-party pretrained weights.
- No preschool capability claim without independent held-out tests for
  coherent sentences, basic factual relationships and simple instructions.
