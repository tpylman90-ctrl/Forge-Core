# Authorized one-off Forge v3 factorized role pilot

Proceed with one bounded experimental run after the PR unit tests pass.

- Original scratch-trained Forge Seed v0.7 weights must remain frozen.
- Two independent random initializations; factorized verified,
  factorized corrupted event labels, question-ablated, and previous
  monolithic role head trained on the same original synthetic examples.
- Maximum **six minutes of CPU research time** and 1800 updates per seed;
  stop earlier at the bounded cap and keep all experiments sequential
  on one hosted Linux runner.
- Evaluate on a newly specified 288-question synthetic exam with disjoint
  names, objects, question forms and source grammars. Previous v2 and v1
  exam prompts cannot be reused as new held-out evidence.
- Separate question-intent accuracy, story-giver accuracy, exact answer
  accuracy and both-correct question-pair performance.
- Preserve a complete report and research-only checkpoint states.
- Do not auto-promote, run continuously, involve personal data, or
  infer a daily GitHub CPU allowance from a successful run.

This is a short feasibility pilot, not another 36-minute stress run.
