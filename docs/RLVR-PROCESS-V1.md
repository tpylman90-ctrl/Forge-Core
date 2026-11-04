# Forge Core — Verifiable Reasoning v1 (RLVR + process supervision)

**Status:** controlled, original-weight-only pilot; NOT merged into the canonical
Forge language checkpoint. It is a tiny standalone neural policy rather than
a language model trained to reason naturally. This separation is deliberate:
Forge Seed's current language proficiency is not sufficient to safely
interpret or use long, free-form reasoning chains.

## Objective

Use **reinforcement learning with verifiable rewards (RLVR)** and
**process-level supervision** to teach a randomly initialized policy to
choose a logically valid intermediate result and verify its own proposed
sequence through an exact oracle.

This is a preparatory mechanism for future native Forge Seed reasoning,
not an assertion of independent thinking or reliable mathematics.

### Formal proof environment

Two original synthetic domains are supported.

**Arithmetic:** a start value and two signed additions. A model selects a
candidate value for each intermediate step. The verifier checks
`proposed == previous + increment` and then checks whether the final
value equals the mathematical target.

**Propositional implication:** a six-proposition finite logic world with
premises and two implications. The model chooses sets of propositions after
each rule application. The verifier independently checks whether the
antecedent is known and whether the proposal exactly matches the correct
logical state transition. An incorrectly applied implication cannot be
rewarded as a valid proof step.

Both domains are small bounded **symbolic environments**. They never run
arbitrary submitted code, shell commands, `eval`, or untrusted executables.
A separately sandboxed, resource-limited code interpreter could be added in
a later security-reviewed phase.

### What the learned model sees

- Task type, numerical input or proposition-state bits.
- Which operation or implication is being applied.
- Two **unlabeled candidate states** (one right and one corrupted).
- Previously selected intermediate state.

It does **not** see the verifier's gold label, correct final answer, or
verified trace as model input. The verifier runs after an action is sampled.

The system records a two-step observable reasoning *trace*. This is an
auditable short chain of explicit proposed states, **not** evidence that
a language model exposes its private internal chain of thought.

### Training conditions

Three equal-budget, from-scratch policy arms:

| Arm | Signal | Purpose |
|---|---|---|
| `outcome_only` | Final-answer reward | Sparse RLVR baseline |
| `process_reward` | Final reward + 0.25 per valid step | Dense verifier-grounded reinforcement |
| `process_supervised` | Same dense rewards + oracle-labeled candidate-step BCE | Step validity as an auxiliary learned process model |

All arms use a stochastic, **on-policy REINFORCE** update with a
within-task group-mean reward baseline. The model policy and the learned
process-validation head share a small encoder. The process classifier is
trained to recognize validity for both proposed choices; this is a
*process-supervised* loss, not a pretrained critic. The oracle remains
independent and authoritative.

A final answer may accidentally match even with wrong steps; it receives
the sparse outcome signal but fails the **complete verified proof** metric.
Only a trajectory with **both** valid transitions and the correct final
state counts as a proved solution. This discourages final-answer-only
reward hacking.

### Evaluation and controls

- Deterministic generation with recorded seeds; original synthetic
  samples do not include personal data or downloaded corpora.
- Held-out arithmetic examples use different numerical start and
  increment ranges; logic examples use unseen random chains and
  occasionally an extra unrelated premise.
- Report exact fully verified proof rate, final-answer rate,
  locally valid steps and separate math versus logic scores.
- Evaluate at identical fixed test seeds for every arm.
- Same randomly initialized policy weights, optimizer and number of
  candidate decisions for all training conditions.
- Versioned, repeatable unit tests include *correct final answer but
  invalid intermediate steps* and a *locally valid second step after
  a corrupted first state*.

### Explicit caveats

- Two-option candidate selection is **not** open-ended language or
  theorem proving. Each two-step task has a correct option by design.
- Synthetic training/exam separation is procedural and not a substitute
  for independent human-authored proofs. Test scores are diagnostic,
  not a formal qualification.
- Early step reward can reward a locally correct second step even if
  the overall proof was corrupted earlier; both process and final
  correctness must be measured.
- Group-relative advantages are a small experimental policy-gradient
  implementation; this is **not** a claim of implementing GRPO/PPO.
- The evaluation corpus is fixed and repeated during this pilot for
  progress visibility, so it is NOT an untouched final certification.
- The trained policy is not Forge Seed; transferring any skill to the
  original pretrained-*from-scratch* language Transformer requires
  a separate benchmark and carefully reviewed integration.
- Never treat generated justifications or model-graded steps as facts.
  Only exact independently checked steps may be used as verified labels.
- No automatic promotion, Android observation, external actions, or
  access to the user's private encrypted memories.

## Execution

Use `python -m forge_core.neural.rlvr_process_v1 --steps 80
--out artifacts/rlvr-process-v1` after installing Forge dependencies.
The dedicated Actions workflow supports a bounded, manually chosen
small CPU pilot; it does not run continuously or seek unlimited GitHub
compute.

The next acceptance gate is **consistent gains on both unseen arithmetic
and symbolic logic** over the sparse-outcome-only control, with stable
proof-level (not merely answer-only) metrics across multiple seeds.
If those results hold, develop a new text-to-symbolic-proof adapter as a
**separate, bounded** branch and test language retention before touching
canonical neural weights.

## Related research

- [Lightman et al., *Let's Verify Step by Step*, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/aca97732e30bcf1303bc22ac3924fd16-Abstract-Conference.html):
  step-level labels versus final-outcome feedback.
- [Wen et al., *Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs*, 2025](https://arxiv.org/abs/2506.14245):
  evaluate correctness of the trace as well as the final answer.
- [Yuan et al., *Verifiable Process Rewards for Agentic Reasoning*, 2026](https://arxiv.org/abs/2605.10325):
  formal per-step oracles provide denser, but task-specific, reinforcement signals.

