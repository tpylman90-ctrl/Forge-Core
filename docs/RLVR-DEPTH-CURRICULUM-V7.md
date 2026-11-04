# Forge RLVR v7 — Short proofs first, then longer formal chains

**Research hypothesis:** Forge's small scratch-started action policy
may be able to learn real, verified 1-step operations before it can
compose 2- or 3-step action sequences. Previous RLVR and supervised
studies failed to reliably generate full three-step proofs, even though
an exact mathematical and symbolic planner succeeds.

## Reference evidence

[Verified v5 result](RLVR-PROOF-GENERATOR-V5-RESULTS.md):
all three action-proposal neural RLVR methods learned to abstain on
every feasible task, even after step rewards.

[Verified v6 result](RLVR-VERIFIED-TRACES-V6-RESULTS.md):
complete teacher-forced formal traces produced **22–29%** correct
non-STOP actions on exact canonical proof prefixes, but only
**1/240** complete shifted proofs across both model seeds.

## Architecture

Retain the same original randomly initialized, independent action
proposal MLP and the **unchanged exact arithmetic/Horn-logic
interpreter**. The five action symbols select arithmetic operations,
logic-rule applications or ABSTAIN. A candidate operation is never
declared correct by the neural model itself: the external pure formal
interpreter checks each proposal and whether a full legal action
trace reached the target.

A new procedural goal constructor uses breadth-first search to
find targets that require **exactly 1, 2 or 3 primitive operations**.
For logical proofs this is the shortest implication depth; for
arithmetic it is the shortest valid operation chain.

The curriculum's teacher trace uses **strict shortest-path labels**:
an action is a valid lesson choice only if it preserves the minimum
number of remaining proof steps. This avoids teaching three-step
detours as though they were one-step examples. Independent formal
execution verifies every selected shortest trace.

Every training phase remains exactly half solvable, half impossible,
and balanced across arithmetic and logic. Impossible cases retain
the ABSTAIN label. All positive process labels are independently
verified before appearing in a loss; no gold answers enter neural
forward inputs.

## Curriculum

- **Stage I:** One-step feasible proofs plus impossible tasks.
- **Stage II:** Mix retained one-step and new two-step proofs.
- **Stage III:** Mix one-, two- and three-step proofs.

A reference control trains on the older **flat 2–3-step**
curriculum for the same number of updates.

Compare three arms from identical random initial weights:
1. `flat_long_proofs`: full verified teacher process labels on
   the earlier all-long curriculum.
2. `staged_proof_imitation`: same objective, new 1→2→3 curriculum.
3. `staged_then_rlvr`: staged supervised curriculum plus a
   small .1-weighted genuine on-policy RLVR term after warm-up.

All receive the same update count, architecture and optimizer, but
not identical training problems; this is a controlled
**curriculum-comparison study**, not proof that RL alone caused gains.

## Results required before promotion

At both development and independently seeded shifted evaluation,
report full verified proof completion and justified refusal
**separately for depth 1, depth 2 and depth 3**, alongside
canonical non-STOP action accuracy and invalid proposals.

Select checkpoints on **development only**, prioritizing genuine
three-step proofs, then two-step proofs and one-step proofs.
The shifted exam contains fresh deterministic seeds with arithmetic
magnitudes outside training and an additional irrelevant logic premise.
It remains the **same internally authored procedural family**,
not an external general reasoning certificate.

Run two random initializations, seeds 5321 and 5322, with one
bounded **900 updates/arm/seed CPU experiment**, no repeating job,
and a 25-minute GitHub Actions timeout. Save research-only best and
last weights, optimizer states, development curves and results.

The goal is to verify a *learning progression*, not to turn the
exact planner into a fake neural achievement. The original Forge Seed
language network, Android companion, and encrypted user memories are
not modified or used.

Even perfect 1-step success would not establish long-form thought,
causality or natural language comprehension. No research checkpoint
is promoted without controls, new external tasks and retention gates.
