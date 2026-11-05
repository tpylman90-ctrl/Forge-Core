# Forge question-intent benchmark v1

## Purpose

Isolate the question-text bottleneck identified by semantic-role experiments. Given only a question, classify whether it asks for the **giver**, **recipient**, or an **other** person/action. The negative `other` class contains ordinary who-questions with no transfer-role assumption.

This benchmark does not test whether a model understood a story or can identify event participants. It is the narrow diagnostic that must transfer before a question-intent head is attached to a story-role model.

## Data design

- 192 training questions: 64 per class, generated from eight manually written wording families and eight training objects.
- 60 development questions: 20 per class, using five different wording families and four disjoint objects.
- 90 final questions: 30 per class, using six additional manually written wording families and five disjoint objects.
- Every record contains only a question, class label, object, and split/family identifier. There is no story context.
- The builder rejects cross-split exact question overlap, wording-family reuse, object overlap, unexpected sizes and class imbalance, then writes split and file SHA256 data.
- No external/pretrained teacher supplies labels. Gold labels are authored with each wording family and are visible in the repository.

## Evaluation plan

1. Freeze the benchmark source and manifest before optimization.
2. Report a majority baseline (1/3), a lexical majority baseline, and per-class confusion.
3. Compare a plain supervised question classifier with a contrastive paired-paraphrase objective using matched examples, update counts and three independent initialization seeds.
4. Select checkpoints using development data only; score final questions once. Report every seed and class, not only an aggregate.
5. Keep the original Forge language checkpoint frozen. Do not attach the question head to story-role inference unless the final score clearly beats controls across seeds.
6. Only after this diagnostic, build a separate end-to-end story-role challenge with independently authored story syntax and new held-out wording.

## Limitations

This is an internal, synthetic, English-only benchmark authored for the project. Disjoint phrasing and object sets reduce template leakage but do not make it an externally standardized test. Public final labels remain visible and must not be reused as an untouched future exam. A successful question-only score would still not demonstrate reading comprehension.


## Integrity run

The dataset-generation and split checks passed in Actions run
[42086273314](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42086273314).
The archived benchmark is
[forge-question-intent-v1](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42086273314/artifacts/13443151662),
with three split SHA256 values in `manifest.json`. An earlier attempt
failed before generation because the workflow had not installed the package;
the setup step was corrected and the successful run regenerated the same
deterministic data.
