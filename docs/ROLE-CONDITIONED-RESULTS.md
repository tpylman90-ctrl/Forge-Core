# Forge Seed — role-conditioned head, completed controlled experiment

**Verified November 3, 2026.** No model promotion.

## Sources and runs

- GitHub Actions research experiment: https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41712121652
- Complete repository CI: https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41712121963
- Frozen original scratch model: run **41572257363**
- Verified compositional research parent: run **41649770774**
- Historical frozen word pointer: run **41670001958**
- New research artifact: `forge-role-conditioned-v1-research`, **13268059936**, expires December 3, 2026 UTC
- Source code and research method: [ROLE-CONDITIONED-V1.md](ROLE-CONDITIONED-V1.md)

## Objective and safeguards

The original randomly initialized Forge Seed language model and its previous
experimental candidate **were not updated**. We trained a small new
question-conditioned entity selector only. Inputs consist of source-detected
capitalized names, their frozen contextual BPE features, limited positional
and preposition indicators, and the frozen question representation.
No verified role label or answer is supplied to the forward pass.

For each arm, **350 updates**, batch **8**, optimizer **AdamW**, seed
**7381**, learning rate **0.001**. The gold and shuffled arms began
with identical new randomly initialized head weights and used the same
example/update schedule. Shuffling across templates **within question role**
preserved the prevalence of each answer position while removing the
grammatical association between syntax and the answer. Training-data
collision handling deterministically resampled duplicate examples, without
exposing seed or gold markers in prompts.

Training: **320** synthetic questions; development: **80**; final
synthetic test: **288**. Names, objects, grammar and question wording
were deliberately disjoint across splits. The final data is internally
authored and not an independently sourced natural-language exam.

## Verified synthetic exam results

| Frozen/new comparator | Correct of 288 | Accuracy |
| --- | ---: | ---: |
| New verified role-conditioned neural head | **162/288** | **56.25%** |
| Matched shuffled-answer head | 148/288 | 51.39% |
| Frozen earlier word pointer, allowed exactly the same two detected people | 150/288 | 52.08% |
| Randomly initialized untrained new head | 144/288 | 50.00% |

The neural head's advantage over the shuffled control is **14 additional
correct of 288**, not enough on this small, correlated synthetic test to
claim robust role comprehension or independent reproducibility.

### Question and counterfactual consistency

| Exact paired criterion | Verified | Shuffled | Earlier pointer, two-name pool | Untrained |
| --- | ---: | ---: | ---: | ---: |
| **Both** giver and recipient questions correct for one event | **20/144** | 10/144 | 9/144 | 3/144 |
| **Both** swapped-person counterfactuals correct | **67/144** | 51/144 | 46/144 | 43/144 |

The primary criterion, both questions correct, remains **13.89%**.
For two independent equiprobable guesses it would nominally be 25%;
the experiment therefore does not demonstrate competent question-conditioned
role binding. The result suggests the model often answers one of the
people regardless of which role the question requests.

By role, verified was giver **77/144** and recipient **85/144**.
The historical head was giver **33/144** versus recipient **117/144**:
a strong answer-position/recipient bias. Both new exam syntax families
remained weak: entrust **80/144**, courtesy **82/144**.

## Research decision: NO PROMOTION

The paired and generalization gates were **not passed**. Do not merge
trained role-head weights into the canonical Forge Seed, deploy them to
Android, call them preschool-capable, or infer natural-language reasoning.
Model source/provenance, controls, and fixed split integrity are verified;
capability is not.

## Next experiment, without exam reuse

1. Isolate **question sensitivity**: does asking who gave versus who
   received actually change the predicted entity? Include a frozen query
   (question-ablated) control.
2. Train **contrastively on paired questions and swapped-role stories**,
   with a matched same-budget per-question training control and within-role
   label-shuffle null. This explicitly penalizes giving the same answer
   to two contradictory role questions.
3. Construct a **new, independently annotated final set** with fresh
   names, sentence families and questions. The 288 questions here are now
   *used* and must never be advertised as untouched in a later trial.
4. Compare the stronger two-slot model to a simple deterministic
   two-person baseline and to the historical pointer on exactly the
   same candidate restriction.
5. Maintain held-out TinyStories language-loss regression monitoring
   when training the backbone is eventually considered; it remains
   frozen in these head-only studies.
6. Repeat promising experiments with independent initialization seeds
   before checkpoint promotion.

**Learning to copy or rank a name is not learning why a person acted.**
The future user-understanding loop requires user observation consent,
explanations, explicit confirmation and encrypted private memory, none
of which belongs in this public model experiment.
