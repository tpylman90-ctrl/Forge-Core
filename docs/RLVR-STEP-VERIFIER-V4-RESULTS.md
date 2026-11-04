# Forge RLVR v4 — canonical-prefix and balanced-verifier study

**Verified result:** both training seeds completed; no capability promotion.

- GitHub Actions: https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41747933263
- Temporary artifact: https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41747933263/artifacts/13284074002
- Merged implementation: PR #11, commit \`098d0e1e45f41f28e5349579cc599de436c088db\`
- Two independent random initialization seeds: **311, 719**
- **100** optimizer updates per arm/seed, same original synthetic
  task generator, new held-out seeds for development/shifted evaluation
- Three arms: ordinary candidate validity BCE, 3:1 positive-weighted
  BCE, 2:1 candidate-prioritized three-action CE with auxiliary BCE
- Both candidate stages learned offline with *canonical verified prefix*;
  this is explicitly **supervised process learning, not a new RLVR result**
- Thresholds selected using development-only balanced safe-refusal/
  feasible-proof harmonic mean; held-out shifted set examined only afterward
- Shifted exam: **240** tasks per seed: 60 feasible proof tasks and
  180 tasks requiring safe rejection. 120 arithmetic, 120 logic per seed.
- Original Forge Seed language model and encrypted personal memories
  remain unchanged. All learned neural policies remain research-only.

## Shifted proof-level results

| Arm | Seed 311 overall /240 | Seed 719 overall /240 | Feasible proofs both seeds /120 | Safe infeasible cases both /360 |
|---|---:|---:|---:|---:|
| Unweighted BCE, learned threshold | **120** | **14** | **12** | **122** |
| Weighted-positive BCE, learned threshold | 68 | 61 | **5** | **124** |
| Direct three-action CE | 120 | 120 | **0** | **240** |
| Always-reject reference | 120 | 120 | **0** | **240** |
| Exact symbolic verifier reference | 240 | 240 | **120** | **360** |

The unweighted head's 14/240 seed **719** result completed
12 feasible proofs but **accepted 224 invalid candidates** and
safely rejected only 2 infeasible cases. This is a serious
safety/coverage failure; the 12 proof successes do not justify use.
The weighted-positive branch accepted **104** and **108** invalid
states for seeds 311 and 719, respectively.

The three-action policy always rejected, reproducing v3's failure.

## Candidate-level validity on shifted rules

| Arm and seed | Positive recall | Negative specificity | Balanced accuracy | Raw accuracy | Brier |
|---|---:| ---: | ---: | ---: | ---: |
| Unweighted 311 | 0% | 100% | **50.0%** | 75.0% | 0.18904 |
| Unweighted 719 | 0% | 100% | **50.0%** | 75.0% | 0.18949 |
| Weighted 311 | 46.25% | 53.19% | **49.72%** | 51.46% | 0.24934 |
| Weighted 719 | 31.25% | 59.44% | **45.35%** | 52.40% | 0.24501 |
| Three-action 311 auxiliary head | 40.83% | 56.53% | **48.68%** | 52.60% | 0.24923 |
| Three-action 719 auxiliary head | 40.83% | 54.31% | **47.57%** | 50.94% | 0.24902 |

The candidate-level test scores **both** proof stages using canonical
prefixes so that early refusal cannot hide positive examples.
Since 25% of candidates are truly valid, the always-invalid reference
has raw accuracy **75%**, positive recall **0%** and balanced accuracy
**50%**. No tested arm exceeds 50% shifted balanced accuracy
consistently across seeds. A constant 25%-valid predictor would have
a theoretical Brier score 0.1875; this pilot did not demonstrate
reliable probability calibration.

Development positive-label performance showed only slight gains and
did not transfer to the larger arithmetic inputs and altered logic
premises. Candidate probabilities, threshold decisions and full
two-step proof correctness are distinct metrics; none can substitute
for the other.

## Interpretation and decision

**DO NOT PROMOTE.** Four related experiments have now verified formal
checkers and different neural training mechanisms while revealing
a consistent representational/generalization weakness on this
small, two-choice, two-step synthetic setting.

The exact logic/arithmetic verifier should remain the authoritative
checker. We should stop spending long CPU sessions trying to make
a neural network approximate these exact small formal rules.

### Next engineering direction

1. Preserve symbolic arithmetic and logical transition functions as
   independent exact oracles.
2. Build a constrained **proof-action proposal** interface: the model
   predicts an operation, premise, or next step *without* seeing a
   guaranteed valid candidate among offers. Reject invalid proposals
   with the exact verifier.
3. Use only accepted, independently checked traces as process labels,
   with a clean distinction between **verified steps** and model
   explanations. No raw internal chain-of-thought attribution.
4. Report the proportion of novel proofs successfully *constructed*
   (not ranked among two supplied candidates), along with number of
   verifier calls, abstention, unsafe proposals, and task transfer.
5. Keep any text-to-formal translation separate until the canonical
   scratch-trained Forge language model demonstrates enough basic
   comprehension. Do not bypass the original-weights requirement.
6. Use independently authored problems or new operation families
   before any capability claim.

This experiment supports **revising the model's proof representation**,
not simply running more optimization steps. Temporary GitHub
artifacts will expire; this ledger preserves the verified findings.
