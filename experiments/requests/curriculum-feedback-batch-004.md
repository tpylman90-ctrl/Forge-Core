# Answer-first feedback curriculum experiment

## Question
Does staging structured feedback improve correction and explanation-field learning when compared with answer-only and one-shot full-record training?

## Conditions
- **Answer only:** supervise the correction only.
- **Full structured:** from update 1, supervise answer, error type, position, observed value, expected value, and repair.
- **Staged feedback:** for updates 1–160, supervise answer plus diagnosis fields; for updates 161–320, add repair.
- The incorrect attempt remains in the input prompt. The target does not spend tokens copying it back.
- Answer is the first output field in both structured conditions.

## Design
Five matched data partitions × three conditions = 15 training jobs. Each model uses the same source-verified public Forge Seed checkpoint, 320 updates, task sampling order, optimizer, rehearsal batches, and shared held-out exam. Score at updates 160 and 320: answer, error type, position, observed value, expected value, repair, complete diagnosis, and complete record. Report paired comparisons and breakdowns by error and task.

No checkpoint promotion; research only.

## Limits
Synthetic three-letter copying only; ten held-out strings produce 100 correlated cases. Exact field scoring is a controlled field-recovery check, not general reasoning or free-form explanation quality. Fresh lineage from the public Forge Seed checkpoint, not the unavailable Session 4 checkpoint.
