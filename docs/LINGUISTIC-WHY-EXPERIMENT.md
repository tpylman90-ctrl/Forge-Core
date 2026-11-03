# Linguistic Why — First controlled neural adaptation experiment

## What is being tested

Can verified, gold-labeled contrasts teach the **original scratch-trained Forge Seed** to choose the right actor, detect standard subject–verb agreement and distinguish affirmative from negated events on unseen nouns and verbs?

This is not conversational curiosity yet. The personal "ask why" loop remains separate, consent-based and local.

The candidate model from [Run 005](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41558008416) has 1,875,072 parameters, trained on the source-content-locked public 800/200 TinyStories preview. It has not achieved coherent free-form text generation. The adaptation experiment starts **two independent copies of those same weights**.

## Experiment groups (three scores, two trained arms)

- **Frozen base:** score the checkpoint directly, with **no** extra training.
- **Gold-label arm:** train on 30 small, original controlled sentence-choice exercises. Each label is derived from a checked grammatical or semantic rule.
- **Shuffled-label control:** same exact prompts, order, number of steps and newly initialized AdamW optimizer, but shuffle labels within each task category while preserving its A/B frequency.

Only the labels differ between the two trained arms. A fixed random schedule picks the same prompt indices for each arm. This is a **null-label control**, not an equivalent amount of general-language training. There are only 30 training and 30 evaluation multiple-choice tasks (five verbs × three probe types, including 20 role choices), so a measured gain could reflect surface-template learning.

## Leakage checks and outcomes

1. **Balanced answers:** agreement and negation answer locations vary; neither always has the correct answer in A.
2. **Separate nouns and verbs:** training and evaluation use disjoint actor nouns AND distinct verbs. Clause and question templates remain shared (a known limitation).
3. **Hidden answers:** the model input includes only a question and answer options; no gold answer or human-authored rule text is in a prompt.
4. **Task-level scoring:** compare forced-choice A/B accuracy and mean negative log-probability for semantic roles, agreement and negation. Chance-level A/B accuracy is approximately 50% on a balanced set.
5. **General-language regression:** evaluate all three networks on the *same existing held-out TinyStories validation split* used for the original checkpoint. The extra training must not quietly destroy ordinary language prediction.
6. **No automatic promotion:** adapted models are kept only as temporary, clearly labeled experimental artifacts. Nothing is published as a working Forge model or installed in Android.
7. **Repeatability:** every run records source checkpoint SHA256, source validation hash, optimizer step/batch/seed, arm scores, training accuracy and evaluation losses.

A single run is a proof of the experiment's validity, not proof of broad grammar understanding. Before making a research claim, repeat across seeds, counterbalanced labels, unseen sentence constructions and an independent annotated corpus. Compare to an equal-compute *text-only* continuation, not just shuffled labels, and report negative results.

## Run

From **Actions → Forge Linguistic Why — controlled A/B experiment**, provide the trusted public training run ID (default: \`41558008416\`) and bounded adaptation steps. Default 120 steps, batch size 2, seed 1234.

The experiment reads only the attributed public dataset checkpoint and original rule fixtures. It neither reads phone data nor touches the personal memory vault.

The first Run 001 is triggered once by \`experiments/requests/why-ab-run001.md\`.

## Interpreting potential outcomes

- Gold arm beats shuffled and frozen on withheld examples **without higher held-out byte loss:** encouraging preliminary evidence for this constrained task.
- Gold arm only excels on training examples: likely memorization / failure to transfer.
- Gold arm wins the multiple-choice task but worsens general-language loss: useful diagnostic of over-specialization; not promotion-worthy.
- No separation from control: revise probes, curriculum or model capacity; do not assume "why" training is always superior.

All findings remain **experimental** and subject to independent replication.
