# Compact structured feedback batch

## Question
Can Forge learn to preserve and use the incorrect attempt when feedback is encoded as short, verifiable fields instead of prose?

## Design
- Five paired datasets; ten total 320-update jobs.
- Each pair uses the same public Forge Seed checkpoint, data split, sample schedule, optimizer, TinyStories rehearsal batches, and sealed 100-case exam.
- Control target: corrected answer only.
- Structured target: attempt, error type, position, observed value, expected value, repair action, and corrected answer.
- At steps 160 and 320, score every field independently and score the whole record only when all fields are exact.
- Compare correction/field accuracy across five data partitions and by error/task category.
- Research only; no checkpoint promotion.

## Limits
The task remains synthetic three-letter copying. Ten held-out sequences yield 100 correlated cases. Exact field scoring measures structured fact recovery, not general reasoning or free-form explanation quality. The run starts from the public Forge Seed checkpoint, not the unavailable Session 4 checkpoint.
