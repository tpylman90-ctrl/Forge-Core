# Evaluation Protocol — initial draft

1. **Integrity:** Run a forward pass; confirm correct token shape, finite next-byte loss and gradients.
2. **Causality:** Changing future tokens must not affect logits for earlier positions (evaluation mode).
3. **Learning:** On a small approved training set, loss must decrease over a controlled fixed-seed run.
4. **Generalization:** Create a held-out set at document level, deduplicate across splits, report validation loss separately from train loss.
5. **Data:** Track dataset URL, revision, license/terms, retrieval date, content filtering and inclusion decision.
6. **Reproducibility:** Record random seed, code revision, model config, device, batch size, optimizer, training steps and checkpoint digest.
7. **Curiosity:** A principle must not move to confirmed state without a user confirmation event.
8. **Authorization:** Every unapproved or mismatched external-action request must be denied by default.
9. **Privacy:** Confirm that public repo and CI artifacts contain no personal memories, keys, tokens or user logs.

Future language benchmarks must be appropriate for a 10–30M parameter scratch model. Do not compare its results directly to production assistants without clearly explaining the capacity and compute differences.
