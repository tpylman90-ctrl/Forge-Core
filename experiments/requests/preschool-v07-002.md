# Forge Seed v0.7 — retry after cross-split document dedup correction

Initial run 41572004117 correctly halted during public-data screening:
some train and validation records in the revision-pinned samples overlapped.
No training or checkpoint mutation occurred.

Approved one-time retry of the same bounded 14,000-step original-model
training request after document-level exact-match removal from the held-out
validation split. The manifest now records dropped overlap documents. Other
source revision, optimizer, tokenizer, training budget, public-only and
privacy constraints are unchanged.
