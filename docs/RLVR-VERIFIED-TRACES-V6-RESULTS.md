# Forge RLVR v6 — complete formally verified trace bootstrapping

**Verified research results. No neural-model promotion.**

- Successful workflow: https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41769831575
- Temporary research artifacts: https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41769831575/artifacts/13294805581
- Merged source `d6b3d34b8ce7ccf570b21a01cd82a227c7896fca`
- Training: **600 updates per arm per independent seed** (5321 and 5322),
  three conditions; **69.75 seconds** study time on a hosted CPU runner.
- New head receives original random weights. **No original Forge Seed
  language weights or Android code were changed.**
- Two- or three-step verified demonstration prefixes were generated
  by the exact oracle; no truth labels enter neural forward inputs.
- Development-only checkpoint selection. New deterministic shifted
  evaluation seeds from the same procedural family; this is
  **not** an externally authored untouched exam.
- Each of two shifted tests contains **120 feasible** and
  **120 impossible** tasks, equally divided across arithmetic and
  finite Horn implication.

## Results: feasible proofs / justified refusal

| Arm | Seed 5321 complete proof /120 | Seed 5322 complete proof /120 | Sound refusals /240 both seeds | Non-STOP action accuracy on canonical shifted prefixes |
| --- | ---: | ---: | ---: | --- |
| Outcome-only RLVR | 0 | 0 | **238/240** | 0% / 0% |
| Complete verified-trace imitation | **1** | 0 | 158/240 | **28.90% / 22.49%** |
| Trace imitation plus RLVR fine-tune | **1** | 0 | 161/240 | **27.75% / 21.89%** |
| Always abstain reference | 0 | 0 | 240/240 | 0% |
| Exact bounded formal planner | 120 | 120 | 240/240 | 100% on valid teacher prefix sets |

The imitation arms learned to attempt non-STOP actions, which the
outcome-only arm did not. However, process supervised success was
**1/240 full feasible proofs across two seeds**, and the hybrid did
not improve on imitation in full proof construction. These small
results are not evidence of robust planning or generalized reasoning.

In one seed, the supervised model stopped soundly on fewer impossible
tasks than the trivial always-abstain baseline; it traded away
correct refusal for almost no verified proof gain. It has **not
passed any promotion gate**.

The canonical-action diagnostic measures action choices on
**teacher-forced exact proof prefixes**, which differ from the
model's self-generated prefixes at deployment. It is an
instruction-sensitivity diagnostic, not a complete proof metric.

## Why longer repetition is not yet justified

- The first action of a three-link implication chain rarely
  yields a final success reward; the remaining choices compound
  error probabilities.
- Verified demonstrations provide partial step-selection signal
  but the neural policy has not learned a reliable multi-step
  program execution algorithm.
- A three-step-only curriculum may be too difficult as an initial
  planning task. The original scratch language model is still
  much earlier in its training trajectory than a capable natural
  language theorem prover.

## Next evidence gate: increasing proof depth

Prepare **new, explicitly verified one-step, two-step and three-step**
arithmetic and implication goals with the SAME 5-action output
vocabulary. First test that the standalone policy learns short
complete proofs, then introduce longer dependencies while preserving
examples of shorter cases. Compare with equal-budget long-proof
training; record both early-stage and full-proof success on
unfamiliar starts and logical premises.

Do not infer a learned independent personality, unrestricted
chain-of-thought, broad proof ability or an Android feature from
these synthetic metrics. Keep exact verifier authoritative and
the original language model separate until independently tested.
