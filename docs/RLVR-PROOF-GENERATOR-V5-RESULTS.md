# Forge RLVR v5 — formal action proposal pilot (verified)

**Status: research-only; no neural weights promoted.**

- Successful GitHub Actions: https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41769105341
- Temporary results/checkpoints: https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41769105341/artifacts/13294500953
- Merged source commit: `10798d1a2a9798b6089a7a6147be8ccb8566f3ef`
- Two independently initialized random policy seeds **5321, 5322**.
- **90** updates per arm/seed, three arms (outcome, process reward,
  additional oracle-labelled process CE).
- **7.93 seconds** experiment time on a standard GitHub Actions CPU
  runner (not an official daily resource allowance).
- Each shifted procedural test: **240** tasks per seed, 120 arithmetic
  and 120 finite Horn-implication problems, equally split between
  solvable and unsolvable cases.
- The neural policy chose from **formal action symbols**, without being
  offered two candidate next states. Exact interpreter verified every
  proposal; bounded exhaustive search generated process annotations.
- Original Forge Seed scratch-trained text model was not loaded or
  modified, nor were Android or private memory files.

## Final shifted results across both seeds

| Model | Solvable proofs completed /240 | Sound abstentions /240 | Correctly resolved /480 |
| --- | ---: | ---: | ---: |
| Outcome-only RLVR | **0** | **240** | 240 (50%) |
| Dense step-reward RLVR | **0** | **240** | 240 (50%) |
| Oracle-process-supervised RLVR | **0** | **240** | 240 (50%) |
| Always abstain reference | 0 | 240 | 240 (50%) |
| Exact bounded planner reference | **240** | **240** | **480 (100%)** |

Random-action reference across seeds: **1/240** completed proofs,
**92/240** sound abstentions. Untrained random-initialized neural
heads varied: **4/240** completed proofs, **93/240** sound abstentions.

Development checkpoints did not solve any of the **60 feasible**
development tasks per seed at update 1, 25, 50, 75 or 90.
All three training regimes collapsed to abstention by update 25.
Development selection used a balanced feasible-proof/sound-abstain
harmonic mean; the gate remained **zero** for all arms. No model
demonstrated novel proof-construction skill.

## Interpretation and engineering decision

**NOT PROMOTED.** Unlike previous candidate-ranking experiments,
the neural policy genuinely proposed formal operations. This rules
out a dependency on supplied correct next-state candidates as the
only problem. However, the exact planner is plainly capable while
the learned policy never constructs a valid complete proof after
training.

Primary credit-assignment failure:
- A single correct first step is insufficient for sparse final
  proof reward, especially for three-link logic chains.
- On-policy process labels are seen mainly on paths the weak policy
  happens to visit, so later steps of successful proofs are absent.
- Sound abstention pays off immediately on half the tasks; always
  abstain is a local optimum, even with a zero balanced acceptance
  score.
- The process-supervised arm saw additional labels but did **not**
  receive full oracle demonstration trajectories or supervision at
  states it never reached.

### Next bounded experiment

Train from **complete, verified canonical action sequences**:
sample a balanced set of solvable and unsolvable problems, supervise
all allowable next actions on true proof prefixes, and then compare
pure supervised imitation with an RLVR fine-tuning arm. Preserve an
outcome-only RLVR reference for clarity, and separately report
training/development/shifted success.

This is a curriculum/bootstrap experiment, not a claim that a small
network can already reason in natural language or that supervised
labels count as learning reasoning autonomously. Do not replace the
original text model; even a positive synthetic benchmark requires
new operations, longer proofs and independently authored problems.
