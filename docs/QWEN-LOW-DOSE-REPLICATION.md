# Forge low-dose Qwen3 language replication

## Design

Replicate the earlier matched uniform-versus-Qwen language study with a
smaller teacher dose. Six arms use the same verified Forge Seed 95k weights,
optimizer moments, tokenizer and pinned TinyStories source. Each training-order
seed (43, 73, 101) has one uniform arm and one 1% Qwen3 target / 99% TinyStories
arm, each for 8,000 updates. The three seeds change sampling order only; they do
not reinitialize the student weights.

Reuse the exact Qwen3-0.6B teacher target artifact from run 42247943454:
revision `c1899de289a04d12100db370d81485cdf75e47ca`, 192 training targets,
32 held-out targets, target-data SHA-256
`cfb60bde877712ca0b8777502a4208becdbde06daef3777ef06ef08524ac312e`.
No user or personal data is included, and Qwen weights remain inference-only.

Each arm compares parent and candidate on 2,048 evenly spaced 256-token
windows from the same TinyStories validation split and independent test range,
and on the same held-out Qwen text. Each also runs the existing 40-orientation
mirrored capability probe, a separate 20-scenario fresh mirrored probe, and
ungraded generation prompts. The fresh probe is generated from a fixed seed;
its prompt hash must match across all arms. Both probes are narrow diagnostics,
not standardized comprehension tests. All checkpoints remain isolated
research candidates.

## Decision rule

Compare Qwen and uniform deltas within each seed. Treat lower loss on synthetic
Qwen held-out targets as evidence of matching those targets only. A useful
distillation signal requires Qwen to improve the independent language result
relative to its paired uniform control across the seeds, without degrading the
capability check. A single low-dose result is not certification or grounds for
Android integration.
