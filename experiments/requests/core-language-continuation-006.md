# Approved Forge Seed v0.7 core language continuation — round 8

Continue from the verified 87,000-step checkpoint in Actions run 42138215923.
Perform 8,000 additional uniform next-token updates, preserving AdamW and RNG
state, tokenizer, architecture, seed 43, and the exact pinned public
train/validation split. Verify parent step, source revision/license, checkpoint
identity, and all data/tokenizer hashes before training.

This corrected request follows a prior workflow attempt that selected the
wrong 68,000-step parent on its push-trigger path. Do not resume that attempt.
Confirm in the job log and lineage.json that this run uses parent run
42138215923 and parent_step 87000 before accepting its candidate.

Measure the 64-window progress curve and 256-window end diagnostic, plus the
fixed matched 2,048-window parent-versus-candidate comparison. Run the bounded
capability check and preserve parent and candidate weights, source split,
metrics, samples, lineage, and matched evaluation in the Actions artifact and
unpublished durable research draft.

Keep this research-only and do not change Android weights automatically.
Stop if the matched held-out language result regresses. A token-loss change is
a text-prediction signal, not proof of comprehension or general reasoning.
