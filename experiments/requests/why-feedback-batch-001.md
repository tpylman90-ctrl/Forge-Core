# Five-pair 320-update feedback robustness batch

## Question

Does adding a verified diagnostic reason and task rule improve correction
quality over answer-only feedback when each method is trained on five distinct,
similar datasets?

## Design

- **Starting model:** source-verified, public Forge Seed checkpoint from run
  `41572257363`. This is a fresh experimental lineage because the local
  Session 4 checkpoint was removed by workspace maintenance.
- **Runs:** five dataset pairs, ten training jobs total.
- **Within each pair:** identical initial weights, 600 lesson examples, sampled
  lesson order, optimizer, 320-update budget, TinyStories rehearsal batches,
  and evaluation prompts. The target differs: answer-only versus reason +
  rule + corrected answer.
- **Dataset partitions:** five disjoint training sets of 60 CVC sequences each;
  one shared held-out exam of 10 previously unseen sequences. Every sequence
  contributes both output formats and all five verified error classes, yielding
  600 training and 100 held-out examples per run.
- **Training:** AdamW, learning rate `0.00012`, batch 4 for feedback, 80% task
  loss + 20% next-token TinyStories rehearsal loss. Score at steps 160 and 320.
- **Measures:** exact correction, exact diagnostic phrase, both together,
  performance by task/error category, and fixed-window validation loss.
- **Selection:** no checkpoint promotion. Compare paired deltas across the five
  data partitions and report mean, standard deviation, and wins/ties/losses.

## Limits

The benchmark uses synthetic three-letter copying tasks. Ten held-out strings
produce 100 correlated cases, so this is an initial robustness study, not a
general-reasoning claim. Exact reason phrase grading does not evaluate whether
free-form explanations are semantically good.
