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
