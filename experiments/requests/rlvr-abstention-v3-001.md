# One-time RLVR v3 controlled abstention benchmark

This request authorizes a **single** research-only pilot on the public
Forge-Core repository, after its new verifiers and unit tests pass.

- Two independent seeds, 311 and 719; 80 updates per arm per seed.
- Compare outcome-only, dense step reward and explicit oracle-process
  supervision on the same original synthetic proof opportunities.
- The action space includes choosing either proposed next state or
  rejecting both if **neither** meets the formal transition rule.
- Evaluate balanced arithmetic and propositional logic cases:
  proof possible, invalid at step 2, invalid at step 1, and both invalid.
- Report overall correct resolution alongside *feasible proof completion*
  and *infeasible safe rejection*. Always-reject and exact-oracle
  baselines must be present to expose trivial success strategies.
- Use new seeds and shifted arithmetic/logic premises, not the previously
  inspected 200-case fixed set. Keep original language checkpoint frozen.
- One CPU runner, 25-minute job cap, no recurring workload. Preserve
  results and experimental optimizer/RNG states as temporary artifacts.
- No pretrained weights, personal memory, arbitrary code execution,
  Android observation, external action or automatic model promotion.

Passing CI or a high rejection rate alone does not establish reasoning.
