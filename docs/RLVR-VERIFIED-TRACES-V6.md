# Forge RLVR v6 — Process-supervise complete verified proof sequences

## Evidence motivating the change

The v5 action-generation policy introduced **actual formal operation
proposals**, rather than choosing supplied candidate next-state values.
The technical pipeline passed CI and CPU evaluation, but all three
trained neural arms converged to always abstain:

- Each of two seeds had 120 feasible proof tasks and 120 impossible
  tasks in the shifted synthetic evaluation.
- Outcome-only RLVR, dense-process RLVR and on-policy oracle process
  supervision each achieved **0/120 feasible proofs**, while soundly
  abstaining on **120/120 impossible tasks** per seed.
- The exact exhaustive formal planner constructed all **120/120**
  feasible proofs and rejected all **120/120** impossible tasks.
- All three learned arms failed the balanced proof/refusal gate.
- The entire v5 CPU study used just **7.93 seconds** of training time.

This is a **credit-assignment problem**: weak on-policy rollouts
almost never visit the second and third steps of a valid proof.
Additional rewards on prematurely terminated rollouts do not give
the actor an example of a complete successful process.

[Verified v5 report](RLVR-PROOF-GENERATOR-V5-RESULTS.md).

## New curriculum: prove first, optimize afterward

The original, formally exact arithmetic and finite-Horn-logic
interpreter remains unchanged. A small original-weight neural
proposal policy selects formal action symbols, not candidate states.

For each training problem, the exhaustive oracle constructs a
**fully verified reference action sequence**. Training replays the
canonical states before every successive action, including step
two and step three. It rewards all action tokens that preserve at
least one complete proof within the remaining horizon.

For impossible problems, the verified label is ABSTAIN at the
initial state. Training deliberately balances solvable and
unsolvable problems, and records both non-abstention action accuracy
and sound refusal accuracy separately.

This is teacher-forced, **process-supervised imitation**. It is
not an indication that the neural network derived the reference
proof itself. The oracle labels influence the loss only, never
the neural policy's forward inputs.

## Three comparison arms

| Arm | Feedback |
| --- | --- |
| `outcome_rlvr` | On-policy sampled terminal correctness, with grouped reward baseline; no teacher traces |
| `verified_trace_imitation` | Complete oracle-verified action demonstrations and set-valued CE |
| `imitation_then_rlvr` | Same verified demonstrations, then a modest on-policy dense-reward RLVR adjustment after warm-up |

All three arms start from the same random weight initialization,
train on the same sequence of formal task inputs and use the same
update count/optimizer family. They **do not** receive equal label
information. The imitation arms are expected to have a major
supervision advantage, which must be acknowledged.

The standalone policy shares **no weights** with Forge Seed's
existing scratch-trained language Transformer or Android app.

## Metrics and controls

Use two independent model initialization seeds, 5321 and 5322.
Default one-shot CPU pilot: **600 updates per arm per seed**, one
serialized hosted CPU runner, job ceiling **25 minutes**.

Every checkpoint is selected by development-only balanced proof
completion and justified refusal. Report:
- Completed valid proofs / feasible tasks on independent new seeds
- Justified abstentions / impossible tasks
- Unverified/illegal proposed actions and number of symbolic checks
- Correct action selection on **canonical valid, non-STOP
  demonstration prefixes**, separately from correct STOP choices
- In-distribution held-out and arithmetic-range / logic-premise
  shifted synthetic evaluations
- Step-by-step development curves and best selected update
- Exact symbolic planner and always-ABSTAIN baselines

The shifted test is procedurally generated from a previously studied
family and **is not an external, sealed exam**. Do not claim
generalization to natural language or mathematical theorem proving.
The next qualifying test must use independent novel operation
families and human-reviewed formal proofs.

## Safeguards

Canonical teacher traces must pass independent formal execution
and full proof validation. All neural action inputs contain only
the problem specification, current state and action symbol—not
verified next-state answers or truth labels.

No downloaded pretrained model weights, arbitrary Python or shell
execution, user data or private memories. All scripts and
checkpoints are isolated to the neural research tree; **do not
modify `android/`** or its GitHub Actions workflow.

The system should not promote research weights to canonical Forge
Seed without consistent new-task improvements, verified reasoning
traces, language retention testing, and user approval at the
architecture integration boundary.
