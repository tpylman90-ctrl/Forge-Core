# Forge Core RLVR v5 — Generate, execute and verify proof actions

**Stage:** standalone original neural reasoning research, not canonical
Forge Seed language training or a claim of open-ended reasoning.

## Why this is the next test

The verified RLVR v3/v4 studies showed that the neural candidate-ranker
repeatedly collapsed to **always reject**, or accepted invalid states.
The exact arithmetic and propositional-logic verifiers were already
correct. In v4, all learned candidate validity models failed to exceed
the nominal 50% balanced validity accuracy baseline consistently on
shifted examples.

Teaching a neural model to approximate the exact checker is therefore
a poor investment. This v5 study changes the neural task from
**select a supplied answer** to **propose an explicit operation**.

## Formal task, inputs and outputs

Two bounded domains, each limited to a maximum of three actions:

- **Arithmetic:** The model receives a starting signed integer and a
  target integer. It emits operations from ADD_ONE, SUB_ONE, DOUBLE and
  HALVE_EXACT. HALVE_EXACT is illegal on odd integers and state values
  are bounded to [-64, 64].
- **Finite Horn-logic implications:** The model receives a set of true
  propositions, four implication rules and a target proposition.
  Action index 0–3 selects the rule to apply. A rule is illegal
  if its antecedent is not known or the consequent is already known.
- **ABSTAIN:** A fifth action declares no proof within three steps.

The model receives only the problem, current state, remaining actions,
and a proposed **action symbol**. It does NOT receive gold action labels,
a correct intermediate state, a solver reachability flag, or a list
of conveniently offered next-state answers in its forward pass.

The independent symbolic interpreter executes each model proposal or
rejects illegal operations. The bounded exhaustive search oracle
determines whether a valid complete proof exists, and which actions
still permit a proof within the residual budget. Its exact labels
are used *after* proposing actions, never as model inputs.

The trace consists of auditably proposed formal operations and
states, not any model's private internal chain of thought. The term
"CoT" in this research means explicit, verified proof trace rather
than revealing hidden internal reasoning tokens.

## New balanced curriculum

The generator makes exactly half of tasks solvable and half
unsolvable, equally split between arithmetic and symbolic implication.
It chooses arithmetic targets based on a bounded breadth-first
enumeration, and generates logic chains with an unreachable decoy
consequent for the unprovable cases.

Three deterministic but separate splits:
- **Train:** initial arithmetic values -7 to +7 and synthetic Horn
  chains using a pool of 8 propositions.
- **Development:** independent seeds with the same value families;
  select best weights using this split only.
- **Shifted examination:** new seeds with arithmetic starting values
  of magnitude 18–28 and an extra unrelated logic premise.

Although shifted, this is the **same synthetic task family**, not
an external formal reasoning examination or proof of generality.
Future tests need independently authored goals, longer dependencies
and new primitive operations.

## Equal-update experimental controls

Each training arm starts from the same original random seed and
newly initialized 64-wide MLP, uses the same formal problems,
optimizer and update count, and chooses actions stochastically.

| Arm | What is learned |
|---|---|
| outcome_rlvr | Group-relative on-policy REINFORCE with reward for a correct completed proof or justified abstention |
| step_reward_rlvr | Same RLVR plus 0.25 per action which remains on a provable path and a penalty for illegal proposals |
| oracle_process_supervised | Dense-reward RLVR plus loss on the **set of all verified actions** that preserve a proof, computed only after action sampling |

The third method receives **strictly more oracle-label information**.
Any gains should be attributed to its information advantage and
architecture, not advertised as a pure reinforcement-learning effect.
This implementation is a small REINFORCE experiment; it does **not**
claim PPO, GRPO or large-language-model RLVR.

Three important references:
- **Always abstain:** should correctly resolve 50% of these balanced
  tasks while constructing 0% of feasible proofs.
- **Random action policy:** uses actual sampled formal actions, not
  a guaranteed correct candidate.
- **Exact bounded planner:** should correctly resolve 100% and
  generate actual symbolic action traces. It is a deterministic
  reference, not a neural benchmark.

An original untrained neural model is a separate baseline.

## Metrics and decision gate

Report **valid complete proof rate among solvable tasks**, **sound
abstention among unsolvable tasks**, total correctly resolved tasks,
illegal proposal counts, and oracle/verifier call counts. Each result
also breaks down arithmetic and logic success.

Select best research weights by **development-only harmonic mean** of
feasible proof rate and justified abstention rate; this gate scores
zero for always abstain. Evaluate the shifted exam once after
checkpoint selection. Report both independent seeds (5321, 5322)
without choosing a lucky seed.

Run a single bounded CPU pilot of **90 optimizer updates per arm per
seed**, with a 25-minute job timeout; do not automatically launch
other jobs or consume unlimited public compute. Preserve best and
last research head weights separately with RNG and optimizer states.
All saved neural components remain experimental and unpromoted.

Success means progress in correctly **constructing** a formal proof,
not merely choosing one of two offered states. A useful model should
beat always-abstain on the balanced gate, maintain low illegal-action
rates, and improve over outcome-only RLVR across both independent
seeds. We must confirm on genuinely new task families before
attaching this to Forge's language model.

## Isolation and safety

This experiment touches only
`src/forge_core/neural/rlvr_proof_generator_v5.py`, its tests,
research documentation and its own workflow. It never touches
`android/`, the Android APK workflow, the original Forge Seed
training checkpoints, memory encryption or external execution.

All primitive operations are pure bounded Python functions.
No user-uploaded code, shell, dynamic Python evaluation or Internet
access is involved. A secure interpreter or text-to-formal adapter
would require a separate design and threat review.

Workflow: `.github/workflows/rlvr-proof-generator-v5.yml`.

Earlier evidence: [v4 negative results](RLVR-STEP-VERIFIER-V4-RESULTS.md).
