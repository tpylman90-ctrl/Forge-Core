# Forge RLVR v4 — Fix the collapsed "always reject" process learner

## Prior evidence

The audited v3 run (Actions **41747102937**) trained two original neural
seeds and three reinforcement conditions. On a 240-task shifted exam,
**every trained actor always rejected both candidates**, resolving only
120 infeasible tasks and zero feasible proofs.

The explicit process head appeared to reach 75% validity accuracy,
but validity prevalence was 25% among first-step candidate labels.
**Predicting "invalid" for everything achieves the same 75%.**

[Read the measured v3 report](RLVR-ABSTENTION-V3-RESULTS.md).

## Design correction

The next pilot must first learn whether each candidate step is valid.
It is **process-supervised verifier training**, not a claim of additional
RLVR algorithmic performance.

Training examples cover both arithmetic and implication logic, both
formal step indices, and both valid-offer and no-valid-offer scenarios.
For stage two, the data generator supplies a **formally verified
canonical prefix**, even if a deployed policy would have rejected the
first offered candidate. This defeats early-rejection data starvation.

The model receives only task/state/candidate features. It does NOT
receive correctness labels or availability flags in its forward pass.
The external exact symbolic checker supplies labels only in the loss.

## Matched small models

All candidates use the same scratch-initialized feature network:

1. **Unweighted BCE:** binary candidate-validity labels as ordinary
   process supervision. Expected to suffer from positive-class scarcity.
2. **Balanced BCE:** the same examples, with positive labels weighted
   3:1 to reflect one valid per four candidate states.
3. **Three-action CE:** explicitly classify offer A, offer B, or
   REJECT BOTH. Choosing a valid candidate is weighted 2:1 over
   rejection. An auxiliary weighted validity loss is also supplied.

All three arms use the same updates, training tasks and optimizer
settings, but not identical loss functions or effective label
weights. Results cannot be attributed to RL alone.

At inference, BCE-trained models choose a candidate only if the
highest learned validity probability exceeds a threshold; otherwise
they reject. The three-action model chooses its highest action logit.

Thresholds for the BCE models are selected **only on new development
problem seeds**, maximizing the harmonic mean of:
- successful feasible proof completion rate, and
- safe resolution rate for infeasible tasks.

That score is zero for any model which rejects everything. The shifted
test is used **after** threshold selection. We also retain raw overall
accuracy and false rejection counts to expose unsafe tradeoffs.

## Required outcome measures

**Candidate-level:**
- Positive-class validity recall
- Negative-class specificity
- Positive precision
- Balanced classification accuracy
- Raw accuracy *alongside* the 75% always-invalid baseline
- Brier calibration loss for validity probabilities

**Proof-level:**
- Overall resolved tasks
- Valid 2-step proofs of the 60 feasible cases out of 240
- Safe rejection of the 180 infeasible cases out of 240
- False rejections and invalid candidate acceptance frequency
- Arithmetic vs symbolic domain breakdown
- New development-range and shifted-range task seeds

Reference baselines remain:
- Always reject: **120/240** overall, **0/60** feasible proofs
- Exact verifier: **240/240** overall, **60/60** feasible proofs

An original, scratch-started neural classifier that only matches
always reject is a failed experiment, regardless of raw validity
accuracy. A model that improves coverage but accepts invalid states
at an unsafe rate is also not ready to integrate.

## CPU and provenance

Two independent training seeds: 311 and 719. Default 100 training
updates per arm and seed, one standard GitHub CPU runner, a 25-minute
job ceiling, serializing related RLVR pilots. Research-only optimizer
states, weights and reports are archived for 30 days.

The shifted data includes larger arithmetic values and an extra
unrelated symbolic fact. This is still the same procedural task
family, not externally authored, independently judged reasoning.
Canonical Forge Seed text weights, its original tokenizer, any
personal encrypted memory, and Android permissions remain untouched.

No arbitrary code interpreter is used or trusted. A secure interpreter,
natural-language to formal-proof conversion and integration into the
original Forge Seed network require separate review and transfer gates.
