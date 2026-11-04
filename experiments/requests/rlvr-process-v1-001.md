# One-off RLVR / process-supervision pilot

Authorized engineering experiment: implement a **self-contained,
verifiably rewarded** two-step arithmetic and propositional-implication
reasoner, starting from randomly initialized neural policy weights.

- Test outcome-only REINFORCE versus dense step-verification reward
  versus dense reward plus explicit process-validity BCE supervision.
- Use equal update budgets, deterministic original synthetic examples,
  and a separate held-out task generator.
- At most **80 training updates per arm** in this one-off launch,
  one standard GitHub-hosted CPU runner, 20-minute job timeout.
- Record final-answer correctness separately from **both valid steps
  and valid final answer**.
- Formal verifiers are local, pure and exact; do not execute user code,
  arbitrary Python, shell scripts or untrusted snippets.
- Keep the original Forge Seed text-model weights and private
  experiential memories completely separate.
- Retain only research metrics and timestamps. No automatic model
  promotion, background monitoring or repeated stress runs.

Successful execution is not proof of general chain-of-thought ability.
It is an infrastructure experiment for training with verifiable rewards.
