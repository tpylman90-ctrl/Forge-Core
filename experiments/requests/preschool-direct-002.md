# Forge v0.7 direct-answer retry — source-pair normalization

The initial direct-answer workflow Run 41642938833 correctly stopped before
training because the two external held-out fixture generators named mirrored
question IDs differently (suffix mirror0/mirror1 versus numeric 0/1). No
model weights or source checkpoints were modified.

This one-time retry validates **adjacent mirrored pair content** directly,
checks that both choices invert while story/question/true answer remain
identical, and treats them as a single natural-answer task.

Original 14k-step source weights, public data, three-stage optimizer
budget, shuffled-label control, and research-only safety conditions are
unchanged. No model promotion or personal-data ingestion.
