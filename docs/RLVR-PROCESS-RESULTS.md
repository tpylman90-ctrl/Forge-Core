# Forge RLVR process-verification ledger

## Experiment 001 — original RLVR and auxiliary process-head pilot

- **Verified:** November 3, 2026 local (November 4 UTC).
- [Successful GitHub Actions run 41744993637](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41744993637)
- [Original temporary research artifact 13282664998](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41744993637/artifacts/13282664998)
- Script: `forge_core.neural.rlvr_process_v1` at merged commit
  `511f51d21791e92db77d8e1aecc39cbf6ca5c2dd`.
- 80 training updates per arm, same random seed **104**.
- 200 held-out original synthetic two-choice tasks, 100 arithmetic
  and 100 six-proposition-implication proof chains.
- Explicit two-step proof traces checked by deterministic formal
  verifiers. A valid final result with any invalid intermediate
  transition is not counted as a fully verified proof.
- No canonical Forge Seed text-model weights loaded or updated.
- The three arms used on-policy REINFORCE with group-relative baselines.
  Process-supervised arm used candidate validity BCE on an auxiliary
  learned head, but **did not directly train the actor from oracle labels**.

| Research arm | Fully verified proof /200 | Correct final answer /200 | Locally valid steps /400 | Math /100 | Logic /100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before training | 52 (26%) | 58 | 208 | 26 | 26 |
| Outcome-only RLVR | **57 (28.5%)** | 62 | 214 | 29 | 28 |
| Dense step reward | 54 (27%) | 57 | 211 | 26 | 28 |
| Step reward + learned validity head | 54 (27%) | 57 | 211 | 26 | 28 |

The measured improvement is **inconclusive/negative** for the
process-reward variants. The process-supervised policy was not better
than sparse outcome-only RLVR. This is still important infrastructure
evidence: the model can generate proposed states, formal oracles can
score each step, the on-policy gradient path works, and synthetic
held-out proof metrics are reproducible.

A task has two options per step and the trained policy was initialized
randomly, so 26% is near the nominal two-choice-by-two-step 25%
reference. The exam is repeatedly observed in our development reports;
**do not treat it as a sealed final certification**. We do not claim a
general chain-of-thought ability, theorem proving, causality or
natural-language comprehension.

### Follow-up decision

Add **direct step-supervised actor learning** alongside verifier
classification and process-conditioned reinforcement. The learned
process checker should also be evaluated as an inference-time
decision maker without the ground-truth oracle. Keep the same sparse
and dense reward baselines, bounded CPU time, no model promotion.
Report both information advantage and checkpoint-level performance.
Do not compare changed process-arm training directly with the old
arm as if the reward algorithms were otherwise identical.

See [mechanism and safety boundaries](RLVR-PROCESS-V1.md).

## Experiment 002 — verifier-to-actor distillation (verified)

- [Successful GitHub Actions run 41745560943](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41745560943).
- [Artifact 13282809191](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41745560943/artifacts/13282809191), temporary research data, 30-day retention.
- Source merge: `465d662bdbf8383abf69480b5b571b7b8139650f`.
- 120 updates per arm, fixed seed 104, same 200 unseen synthetic tasks.
- Change from experiment 001: process-supervised arm receives **direct
  oracle-labeled actor choice CE (coefficient 0.65)** plus auxiliary
  process-validity BCE (coefficient 0.25), in addition to its
  grouped-baseline RLVR policy-gradient term. Both other arms retain
  the earlier reward definitions.
- Process validity labels are calculated by an exact verifier after
  generating candidate actions; **no gold labels are passed to
  network.forward()** or the greedy evaluation chooser.
- The process head may independently choose actions at inference.
  Both actor-driven and process-head-driven results are preserved.

| Arm | Full actor-verified proofs /200 | Candidate validity classifier /800 | Full proofs when process head selects actions /200 |
| --- | ---: | ---: | ---: |
| Untrained actor | 52 (26%) | 400 (50%) | Not recorded |
| Final-answer-only RLVR | **51 (25.5%)** | 398 (49.8%) | 52 |
| Dense process rewards | 51 (25.5%) | 382 (47.8%) | 52 |
| Verifier-supervised actor and self-check | **45 (22.5%)** | **443 (55.4%)** | 45 |

Detailed domain breakdown of fully verified actor proofs:
- Outcome-only: math **23/100**, logic **28/100**.
- Process-reward-only: math **23/100**, logic **28/100**.
- Verifier-supervised: math **23/100**, logic **22/100**.

The step checker is a few percentage points above a nominal 50%
two-class reference on these examples, but this does not translate into
better decision-making, and it does not beat the sparse-reward arm.
The 200 problems come from a single synthetic held-out generator and
have been inspected in development, not certified as an untouched
qualification exam. More seeds and independent distributions are
needed before statistically interpreting differences.

**Decision: do not promote any RLVR head.** Two real GitHub experiments
validate the training/verification machinery but **not a useful reasoning
capability gain**. The original Forge Seed language network remains
unchanged, and no pretrained model weights or personal memories are
mixed into this research.

### Next diagnostic

1. Split performance into **in-distribution**, novel arithmetic-number
   ranges, and novel symbolic-premise distributions. Isolate whether
   the failures are task representation, sparse credit, or generalization.
2. Use a *new* seed and an untouched held-out challenge for the next
   experiment; do not continuously optimize against this same
   200-case sample.
3. Evaluate actor confidence versus process-head confidence under
   calibration, and include **verifier abstention/rejection** when no
   valid candidate is available.
4. Compare a simple non-neural finite-state verifier baseline to
   learned policies. Avoid granting the neural model artificial credit
   for symbolic rules that a deterministic checker can already solve.
5. Only after consistent control-beating results consider a
   **text-to-symbolic-proof** interface for Forge Seed, with
   separate language-retention testing and no raw hidden CoT claims.
