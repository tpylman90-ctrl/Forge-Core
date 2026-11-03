# Forge v0.7 — Learning the Relationship, Not Just the Prompt

**Research target:** Friday, November 6, 2026. The model is not yet
preschool-capable; the goal remains measured improvements in vocabulary,
sentence understanding and short-instruction following.

## Verified starting point

The original 2.22M-parameter model learned to write recognizable,
imperfect TinyStories-like passages. Our first direct-answer text
apprenticeship reached **256/256** on practiced examples, but only
**9/20** on a familiar-template held-out test and **6/20** on unseen
names and phrasing (against **4/20** and **3/20** for shuffled training).

Earlier A/B lessons did not generalize either. More repetition of narrow
templates alone is not sufficient evidence of linguistic understanding.

## Next training experiment

Generate *original* verified semantic facts across five skills:
color, giver, recipient, location, and count. For each, vary the
people, objects, fact order, verbs, question wording, and which entity
the question targets. Rather than guessing an answer letter, Forge
must generate the correct answer word or person's name.

The teacher is a deterministic generator with explicit truth labels.
It is not another pretrained language model. It does not ask Forge
to assess the correctness of its own answers.

Train two research arms from the exact same audited original checkpoint.
One sees verified answers; the other sees shuffled answers within each
skill. Both get identical public-language rehearsal and optimizer steps.

| Stage | Training questions | Updates per arm | Public language rehearsal |
|---|---:|---:|---:|
| Early grounding | 160 | 250 | 5% |
| Varied practice | 800 | 650 | 15% |
| Broad combinations | 3,200 | 1,600 | 25% |

## Evaluation and release gates

Build a fresh **200-question final challenge** with 40 cases per skill,
using names and objects that do not appear in the training generator
and separate sentence/question template families. Use a *distinct*
50-question progress set only for intermediate engineering diagnostics.
No final challenge answers are used for training or stage selection.

Evaluate unmodified original, verified-answer and shuffled-answer models.
Measure exact greedy short-answer accuracy overall and per skill,
general held-out TinyStories token loss, and fixed-prompt story samples.

This remains an internally authored synthetic benchmark. Even a strong
result must be replicated on separately sourced natural-language tasks,
more seeds and free-form one-step instructions. Never automatically
promote experimental weights or claim preschool-level comprehension.

Current archived original pretraining run:
https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41572257363

Last verified direct-answer pilot:
https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41643271245
