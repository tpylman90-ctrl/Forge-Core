# One-time Forge RLVR v6 verified-trace bootstrap

Research objective: test whether **complete formally verified action
demonstrations** overcome the persistent always-ABSTAIN failure in v3,
v4 and v5.

- Do not modify Android or original Forge Seed language weights.
- Use a new randomly initialized small formal proof-action policy.
- Maintain the same arithmetic and finite implication oracle,
  original synthetic data generator and exact verification.
- Train three arms on identical original synthetic problems:
  outcome-only RLVR, complete proof-action imitation, and imitation
  with modest policy-gradient optimization.
- Two seeds 5321 and 5322; **600 optimizer updates per arm per seed**;
  maximum 25-minute GitHub-hosted CPU job, serializing RLVR pilots.
- The imitation arms receive more oracle labels than pure RLVR;
  disclose that information advantage instead of calling it
  unsupervised reasoning.
- Select checkpoints on development data only; evaluate distinct
  shifted procedural seeds after training, and report both success
  and failure.
- Include exact-planner, always-ABSTAIN and separate verified
  non-STOP action accuracy baselines to defeat refusal shortcuts.
- Save temporary experiment/checkpoint artifacts; **never promote**
  research heads or claims of general language comprehension.

This is one bounded, non-recurring experiment; no further training
job should be triggered by this request after its initial merge.
