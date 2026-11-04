# Forge RLVR v3 — abstention under distribution shift (verified)

**Research only. No canonical language-weight or experimental-policy promotion.**

- [Successful GitHub Actions run 41747102937](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41747102937)
- [Temporary experiment artifact 13283828294](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41747102937/artifacts/13283828294)
- Merged code commit `4f7a5459ae5e55cad67318084cb32f1a1a1cc69b`
- Two independent train seeds **311, 719**; three separate arms
- **80** optimizer updates per arm, best selected on development data (both arms selected step **20**)
- Shifted synthetic exam **240** tasks per seed, 120 arithmetic and 120 logic,
  with more difficult arithmetic ranges and an extra unrelated logic premise
- Balanced offer scenarios: **60** VV (complete feasible proofs),
  **60** VN, **60** NV, **60** NN
- An explicit third action **REJECT BOTH**; trained neural policy and learned
  step-validity head started with random weights. Original Forge Seed text model
  not loaded or updated.

## Verified outcome, on BOTH seeds

| Model | Correctly resolved /240 | Verified feasible proofs /60 | Safe rejection on infeasible /180 | Invalid accepts | False rejections |
|---|---:|---:|---:|---:|---:|
| Random initial, seed 311 | 15 | 9 | 6 | 212 | varying |
| Random initial, seed 719 | 15 | 15 | 0 | 225 | varying |
| Always reject | **120** | **0** | **120** | **0** | **120** |
| Exact symbolic verifier reference | **240** | **60** | **180** | **0** | **0** |
| Outcome-only RLVR | **120** | **0** | **120** | **0** | **120** |
| Dense process RLVR | **120** | **0** | **120** | **0** | **120** |
| Explicit process supervision | **120** | **0** | **120** | **0** | **120** |

All three trained policies converged to **always reject** under both
independent seeds, on the development and shifted synthetic splits.

The learned process validity head reached **75% label accuracy** in the
process-supervised arm on both seeds, but only one of four evaluated
candidate labels is positive, because the model stops after step 1.
Consequently **always predicting invalid is a 75% majority-label
baseline**. The process head's observed Brier scores were approximately
**0.2059** and **0.2135**, while a constant 25%-positive probability
has a theoretical Brier score of **0.1875** under this class balance.
This does not show meaningful validity discrimination.

The dense-reward and sparse-reward arms showed validity accuracies
**25–60%** depending on seed, but every actor refused all offers.
The original exact formal oracle still works; its perfect accuracy is
a rule-based reference, not a learned reasoning capability.

## Root cause diagnosed

- Correct refusal is rewarded for both NV and NN cases, and complete
  proof requires **two** consecutive valid selections. "Always refuse"
  provides an easy 50%-resolution local optimum.
- The rollout ends at rejection, so a policy collapsing early receives
  essentially **no second-step training data**.
- Candidate validity classification is highly imbalanced (one positive
  per four offered candidate labels), making raw accuracy misleading.
- Development-only checkpoint selection optimized overall resolution
  before feasible proof coverage, reinforcing this collapse.
- Oracle-derived process labels were used in losses, not model inputs;
  the performance failure is not a successful proof of cheating.

## Research decision

**NO PROMOTION.** Do not extend training time on this flawed reward
structure. Correct the experiment before spending another long CPU run:

1. Train step recognition on **both proof steps using verified canonical
   prefixes**, whether or not the policy would have rejected earlier.
2. Compare unweighted validity BCE against imbalance-aware BCE and
   directly supervised three-action classification, with matched tasks.
3. Select decision thresholds on development only with a **balanced**
   feasible-proof and safe-refusal objective. Always-reject must score zero
   on the feasible-proof component.
4. Evaluate positive-class recall, negative specificity, precision,
   balanced accuracy, Brier, invalid accepts, and per-domain proof
   completion; compare with 50% always-reject and exact-oracle baselines.
5. Use new task seeds and independent random initializations. The v3
   shifted exam has now been inspected and is not an untouched benchmark.
6. Keep the explicit, pure formal verifier authoritative; any neural
   checker is an experimental assistant to verification, not a substitute.

See [mechanism](RLVR-ABSTENTION-V3.md).
