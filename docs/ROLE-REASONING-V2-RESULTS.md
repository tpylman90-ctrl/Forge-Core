# Structured Role-Explanation Study v2 — Results

**Completed:** 2026-11-05  
**Actions run:** [42108171047](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42108171047)  
**Artifact:** `forge-role-reasoning-v2`, expires 2027-02-03  
**Artifact SHA256:** `33bc310151fab4703efa41270acae18c1666db90b8262857b47dfa3b00b06767  
**Decision:** No promotion; no Android integration

## Question

Does a deterministically verified giver/recipient explanation help the
36,000-step Forge Seed checkpoint identify roles, and does requiring a
candidate letter separate role selection from spelling an unfamiliar name?

## Design

Three independent seeds (101, 202, 303) trained four equal-update arms from
the same frozen source checkpoint: structured explanation plus exact name,
structured explanation plus A/B choice, within-role shuffled labels, and
text-only continuation. Each seed had its own 3,200-row training set and
disjoint name pool. All arms saw the same 240-row final exam for that seed;
final names, objects and templates were held out, with unseen transfer verbs
in half the rows. Gold labels were generated and checked by deterministic
program logic. No external AI teacher or private data was used.

The total run took **4,018 seconds** (about 67 minutes). Final metrics were
not used for checkpoint selection.

## Results

| Arm / metric | Aggregate | Seed 101 | Seed 202 | Seed 303 |
|---|---:|---:|---:|---:|
| Structured explanation, candidate letter | **387/720 (53.75%)** | 146/240 (60.8%) | 135/240 (56.3%) | 106/240 (44.2%) |
| Structured explanation, exact name | **1/720 (0.14%)** | 0/240 | 1/240 | 0/240 |
| Frozen source, exact name | **0/720** | 0/240 | 0/240 | 0/240 |
| Shuffled-label control, exact name | **0/720** | 0/240 | 0/240 | 0/240 |
| Equal-compute text-only control, exact name | **0/720** | 0/240 | 0/240 | 0/240 |
| Both giver and recipient trace correct for a whole episode | **0/180** | 0/60 | 0/60 | 0/60 |

The letter arm averaged only 3.75 percentage points above 50% chance, changed
substantially across seeds, and failed to produce one fully correct role trace
for any held-out episode. Exact-name generation did not learn the task.

The structured-name arm's held-out language loss after adaptation was
2.377515, 2.372432 and 2.361520 for seeds 101, 202 and 303, respectively,
against the frozen source value of 2.369578. Retention therefore varied by
seed, with two small regressions; the third improved. These short runs do not
establish a reliable language-retention effect.

## Interpretation

This result does **not** show reliable role understanding. Candidate-letter
accuracy alone is vulnerable to format and position strategies; the trace
metric and seed variation reject a positive claim. The result supports stopping
this exact structured-explanation recipe rather than repeating it unchanged.

This is an internally authored synthetic task, not an independent natural
language benchmark. A structured answer string is not proof of hidden internal
reasoning. No candidate weights were promoted.

## Next step

Continue approved next-token learning on Forge Seed's original core weights,
which is tracked in separate [36,000-step](CORE-LANGUAGE-CONTINUATION-36000.md)
and [44,000-step](CORE-LANGUAGE-CONTINUATION-44000.md) reports. For role
learning, redesign the task/architecture using these failure modes, then test
a fresh sealed evaluation before spending another long run. Keep language
training and role-learning research as distinct evidence streams.
