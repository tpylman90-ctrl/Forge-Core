# Approved three-seed low-dose Qwen language replication

Run the six-arm matched comparison described in `docs/QWEN-LOW-DOSE-REPLICATION.md`:
uniform versus 1% Qwen3-generated language targets at training-order seeds 43,
73, and 101. All arms start from Forge's exact verified 95k scratch-origin
checkpoint, resume its optimizer moments, keep the Forge tokenizer and pinned
TinyStories data, and run exactly 8,000 updates.

Reuse the exact Qwen3 target-data artifact and hash from run 42247943454; do not
regenerate or alter its prompts, examples, model revision, or split. Qwen model
weights remain inference-only. No personal data is allowed.

For every arm, compare parent/candidate on the same 2,048-window, 256-token
validation and independent test evaluation, held-out Qwen text, the existing
mirrored capability suite, and the separate fixed-seed fresh mirrored probe.
Archive all target text, manifests, hashes, optimizer state, checkpoints,
metrics, capability results, generated samples, and lineage. Mark candidates
research-only; no Android promotion or integration.
