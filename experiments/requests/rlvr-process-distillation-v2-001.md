# One-time RLVR v2 verifier-to-actor feedback run

- Train outcome-only RLVR, final-plus-dense-step RLVR, and
  **verifier-supervised actor + process classifier** branches.
- Same random initialization, synthetic task generator and update
  schedule. The third branch receives additional correct-step oracle
  labels; make the unequal information explicit in the report.
- Record correctness of both explicit proof steps and final answer,
  process-head validity classification, and verified proof success when
  the self-check head alone selects candidate states.
- 120 updates per branch, one GitHub-hosted standard CPU runner,
  maximum 20-minute job. Do not run repetitions automatically.
- Save results; maintain original scratch-trained Forge language
  checkpoint and private encrypted memories completely unchanged.
- No arbitrary code execution, auto-promotion or public
  claim of natural-language chain-of-thought.
