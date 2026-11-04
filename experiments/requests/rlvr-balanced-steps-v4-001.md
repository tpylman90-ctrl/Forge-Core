# Forge v4 — one-off class-balanced formal verifier experiment

Authorization: one bounded follow-up study to correct the measured
"always reject" collapse in the v3 RLVR research head.

- Begin from a newly random-initialized, stand-alone proof-step network,
  not pretrained language or previously experimental proof weights.
- Train on independently verified canonical prefixes for **both**
  stages even when a live policy would have rejected stage one.
- Compare unweighted validity BCE, 3:1 positive-weighted BCE, and
  balanced three-action CE including REJECT BOTH.
- Two independent train seeds (311 and 719), 100 training updates per
  arm, one standard GitHub CPU runner, job timeout 25 minutes.
- Select rejection thresholds exclusively on the development seed.
- Score candidate positive recall/specificity/balanced accuracy and
  feasible proof completion/unsafe acceptance on new shifted seeds.
- Always-invalid (75% raw validity), always-reject (50% overall) and
  exact-verifier (100%) baselines must all remain explicit.
- Archive temporary research-only weights, optimizer states and
  experimental results; no canonical checkpoint promotion, Android
  integration, personal memory use or arbitrary code execution.

Passing tests or achieving high raw validity accuracy alone does not
qualify as learned reasoning.
