# Preserving Forge Seed v0.7 (research only)

The original 14,000-step v0.7 language model was trained on public
TinyStories story excerpts from a pinned upstream dataset revision. Its
2,219,136 neural parameters and its 2,048-token BPE tokenizer were learned
from scratch; no pretrained-model weights were copied.

The GitHub Actions training artifact is temporary. The safe archive
workflow creates an **unpublished DRAFT GitHub Release**, after verifying
the SHA256 hashes of the public text, tokenizer, manifest, checkpoint,
optimizer/RNG presence and finite held-out loss. It never overwrites a
previous tag, never uploads private memory and does not certify language
understanding.

To archive a later public-only experiment, choose Actions →
**Forge Seed v0.7 — audited draft research archive**, supply the successful
pretraining run ID, a unique tag and select the explicit
approve-public-research-draft control. Then review the draft release
manually before any public publication. Private memory must NEVER be
stored in GitHub Releases.

The first safe draft archive is requested once by
experiments/requests/archive-v07-001.md, using training run 41572257363.
The unrelated question-answer teaching candidates remain research only
and are not included in the baseline archive.
