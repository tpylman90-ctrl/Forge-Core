# Forge RLVR v3 — When no reasoning step is valid

**Status:** original research policy, not canonical Forge Seed neural weights.

## Motivation

The first two verifiable-process pilots proved that the training and
formal-checking machinery works, but **did not** improve full proof
accuracy over final-answer-only reinforcement learning. On the second
pilot, the process checker classified **443/800** candidate states
correctly while the actor fully verified **45/200** proofs, compared
with **51/200** for the outcome-only policy.

Both earlier environments *guaranteed* that exactly one of two offered
states was correct. This made "always select a candidate" an artificial
shortcut: the model had no opportunity to acknowledge missing evidence.
V3 makes **REJECT / NONE VALID** a first-class action.

## Formal tasks and independent oracle

The original two-step arithmetic and propositional-implication
environments are reused, but each step offers two distinct candidates
with either **one** correct state or **zero** correct states. The
independent verifier checks each proposal against the exact transition.

A candidate is valid only if it matches the precise transition
\`expected_step(task, step, previous_state)\`. Neither availability nor
the verifier's answer is supplied to the network; proposals are
unordered and their order is randomized.

The action space is:
- \`0\`: accept first proposed next state
- \`1\`: accept second proposed next state
- \`2\`: reject **both** proposed next states

Rejecting both is correct only if both fail the formal checker.
Accepting an invalid transition always fails the episode immediately.

Four offer regimes are balanced *within both arithmetic and logic*:
- VV: one valid candidate at both steps
- VN: first step valid; second step has none
- NV: first step has none; second would be valid if reached
- NN: neither step has a valid candidate

One-fourth of tasks can be completed as a full valid proof.
The others must safely stop at the first step lacking a valid proposal.
Therefore an **always-reject baseline resolves 50%** of tasks safely
while completing **0%** of feasible proofs; a neural system must
improve both safety and successful completion. The exact reference
verifier resolves **100%** by definition (not a learned baseline).

## Trainable architecture

New independent original-weight-only neural network:

- 32-feature candidate vectors from the task, state and proposed result
- Shared small randomly initialized candidate encoder
- Two choice logits and a permutation-invariant third reject logit
- Separate learned step-validity head

All neural weights start random. No pretrained teacher, Android
observation, personal memory, or free-form language proof is used.

Three conditions receive the same random initialization, optimizer
configuration, task schedule, and update count:

1. **Outcome-only RLVR:** on-policy score-function policy gradient,
   with grouped per-task relative reward baseline, receiving only
   complete proof/safe-rejection end reward.
2. **Dense process RLVR:** the same objective plus correctness feedback
   for each action and valid rejection decision.
3. **Explicit process supervision:** dense rewards plus exact
   action-choice cross-entropy and per-candidate validity BCE.
   This third arm receives strictly **more oracle label information**;
   performance cannot be attributed solely to RL.

## Evaluation

Run two independent initialization seeds **311, 719**.
Bound the default CPU pilot to **80 updates per arm per seed** on one
standard GitHub Actions runner; 25-minute job timeout, no recurring
scheduled runs or parallel stress tests.

The task generator has three distinct reproducible splits:

- **Train:** small signed additions and two-rule implication chains.
- **Development:** unrelated problem seeds, same mathematical ranges,
  used for checkpoint selection.
- **Shifted:** larger arithmetic initial states and operation increments,
  plus an extra unrelated symbolic premise. New task seeds, not the
  previous 200 evaluated questions.

The shifted data still uses an existing procedural task family,
so this is **not** an external, sealed reasoning benchmark.

Every scored run reports:
- Exactly completed valid two-step proofs **only among feasible tasks**
- Safe-rejection resolution **only among infeasible tasks**
- Overall resolution, distinguishing acceptance from refusal
- False rejections and invalid candidate acceptances
- Invalid-accept risk among accepted actions
- Learned process-validity label accuracy and Brier score
- Per-domain and per-offer-regime breakdowns

Record random-initialized, always-reject and exact-oracle baselines.
The exact-oracle benchmark is 100% by construction and does not
demonstrate a learned capability.

The pilot preserves research-only best-development weights, final
optimizer weights/state, CPU RNG states and metrics separately. No
checkpoint is promoted; experiment artifacts expire under GitHub's
retention rules.

## Gates

We will not claim meaningful learning unless the neural policy
beats always-reject on *overall* resolution while also completing
a meaningful fraction of feasible proofs, limits invalid acceptance,
and transfers to the shifted family. Results across two seeds must
be compared rather than cherry-picked.

A clean proof checker always remains authoritative. Forge must never
use confidence in a neural step checker as a replacement for a formal
verifier where the latter is available.

The next stage would be a new independently authored multi-step
formal environment. A secure code execution sandbox and any
integration with Forge Seed's original scratch language network
remain separate future security/retention work.
