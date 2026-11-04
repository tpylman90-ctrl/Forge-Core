# One-off English Class v1 read-only retention audit

The first genuine six-level English Class training run
**41775152111** succeeded and archived all three research-only
models as artifact **13296023838**.

The grammar results are valid for the internally generated 96-question
synthetic exam: frozen 48/96, ordered class 48/96, mixed lesson order
53/96, corrupted-label control 47/96. The 53/96 improvement is
small and not statistically conclusive.

However, the original program accidentally scored frozen-model
public-story loss using 32 fixed windows and adapted-model loss
using 48. Those original reported retention ratios **must not be
treated as directly comparable**.

Authorized follow-up: one **read-only** exact-source audit. Download
the archived original Forge Seed model and existing experimental
training results, verify their SHA256 lineage, and recompute the
original and every adapted model's TinyStories validation loss on
the **same 48 fixed windows of 64 tokens**. Re-score the already
inspected English exam only as an integrity check, not a fresh
held-out proof of generalization.

One short single CPU GitHub Actions job, 18-minute timeout,
no neural training updates or model promotion. Preserve corrected
numeric retention metrics without re-uploading large weights.
Do not modify Android or any user-private data.
