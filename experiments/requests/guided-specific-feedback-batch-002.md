# Attempt-specific guided feedback batch

## Question
Does explicit, attempt-specific diagnosis and repair help Forge correct mistakes while preserving correction accuracy, compared with answer-only supervision?

## Design
- Five paired data partitions; ten total training jobs.
- Each pair uses the same verified public Forge Seed checkpoint, 600 lessons from 60 sequences, optimizer, sample order, rehearsal batches, 320 updates, and sealed 100-case exam.
- Answer-only target: exact corrected answer.
- Guided target: identify the exact missing/extra/substituted letter, ordering mismatch, or formatting error; state the repair; then provide the exact corrected answer.
- Score exact correction, exact diagnosis+repair, both together, performance by error/task, and fixed-window text loss.
- No checkpoint promotion.

## Relation to the previous batch
Data partitions, starting checkpoint, update count, optimizer, and evaluation prompts are held constant. The earlier explain-why branch used broad error labels and general task rules; this branch names the offending letter or position and gives an explicit repair. This allows a direct comparison with both the repeated answer-only control and the earlier generic explanation results.

## Limits
The task remains synthetic three-letter copying. Ten held-out strings yield 100 correlated cases. Exact phrase scoring is strict and does not measure free-form explanation quality or general reasoning.
