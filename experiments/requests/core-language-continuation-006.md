# Approved Forge Seed v0.7 adaptive core language continuation — round 8

Continue from the verified 87,000-step checkpoint in Actions run 42138215923.
Perform up to 8,000 additional update attempts, preserving the exact AdamW and
RNG state, tokenizer, architecture, seed 43, and pinned public train/validation
split. Verify parent step, source revision/license, checkpoint identity, and all
data/tokenizer hashes before training. Do not resume the incorrect 68,000-step
parent from the earlier workflow attempt.

Save a resumable optimizer/model/RNG checkpoint every 250 updates. Evaluate on
the same fixed 64-window validation sample at each checkpoint. If validation
loss exceeds the best checkpoint by more than 1% for two consecutive checks,
automatically roll back model, optimizer, and RNG to the best checkpoint, halve
the learning rate, and continue. Reduce no lower than 0.00001; if another
sustained regression occurs at the floor, stop. Keep both best and latest
checkpoints, curve, and recovery events.

At completion, compare parent and best candidate on the same 2,048 validation
windows. Preserve artifacts and mark the result rejected if matched loss is
worse; do not create a durable draft in that case. Run the bounded capability
check only after the matched gate passes. A candidate remains research-only:
do not change Android weights or promote it automatically.

A token-loss change is a text-prediction signal, not proof of comprehension or
general reasoning.
