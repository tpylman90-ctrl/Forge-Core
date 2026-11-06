# Approved bounded Qwen3 language-distillation pilot

Run the isolated Qwen3-0.6B sequence-distillation pilot described in
`docs/QWEN-DISTILLATION-V1.md`.

Use only the pinned Qwen repository revision and non-personal synthetic prompts.
Keep Qwen weights inference-only. Start both the Qwen-mixed arm and uniform
control from the exact verified Forge Seed 95k checkpoint
`42143390701` (SHA-256
`a7bcdb996e440f44d7496b887b54d19324927b4296fe9ed2f0f0293c88c63a0a`).
Resume its optimizer and RNG in both arms, use identical Forge tokenizer and
source splits, and run 8,000 updates per arm. Qwen arm uses 10% generated text
and 90% original TinyStories training sequences.

Preserve Qwen target text and prompt split, teacher revision/license, all data
hashes, parent and candidate checkpoints, optimizer state, evaluation reports,
capability results, and lineage in artifacts and draft releases. Compare both
arms against the same 2,048-window, 256-token validation and independent test
sets, plus held-out Qwen text and the existing mirrored capability probe.
Mark every candidate research-only; do not promote or integrate into Android.
