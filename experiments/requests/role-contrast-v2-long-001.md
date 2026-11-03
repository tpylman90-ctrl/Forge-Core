# Authorized one-off research run: contrastive role learning v2

Owner authorizes a **single** longer Forge Seed controlled training
experiment using the existing free standard GitHub-hosted CPU runner.

- Train **three independent research-only** initialization seeds,
  each with four matched conditions: paired contrastive, ordinary
  cross-entropy, question-ablated contrastive, group-flipped labels.
- Hard bound to **36 minutes of training** and a 48-minute workflow
  timeout for setup, reporting and shutdown.
- Run on **one runner at a time**. Do not implement automated
  repeated stress runs or attempt to bypass usage policies.
- Use only existing audited Forge Seed scratch checkpoint and original
  synthetic curriculum. New final exam split must exclude all prior
  exam vocabulary, grammar families and prompts.
- Report steps/second, walltime, dev-vs-final exact match, paired
  role consistency, counterfactual consistency and controls.
- Save temporary experimental snapshots and metrics; do not promote
  model weights or use personal data. Keep original Forge Seed frozen.
- Treat this request-file push to main as the single experimental
  trigger, not as authorization for unbounded daily consumption.

Success means a measured, reproducible answer about this architecture,
including negative results. It does NOT imply general preschool
comprehension or a known GitHub daily compute entitlement.
