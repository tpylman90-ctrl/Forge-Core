# Forge v0.7 — Copy and role understanding

After 14,000 steps of original TinyStories language pretraining, followed
by a 3,200-example synthetic direct-answer apprenticeship, Forge Seed
reached **49/200** on a fresh internally authored QA exam, vs **13/200**
for shuffled-target teaching. However, it correctly answered **0/40**
giver and **0/40** recipient questions featuring unfamiliar people.
Its independent one-step instructions assessment reached **11/50**,
with no correct word-copying or list-selection answers.

A small causal Transformer may learn a particular answer vocabulary
without learning the general process of copying a referenced name.
That is the next training hypothesis—not an established diagnosis.

## Experiment

Branch two independent copies from the **verified compositional
research candidate** (not from the historical original checkpoint)
after verifying exact archived source weights and matching tokenizer.
Train deterministic original commands for copying words and names,
selecting requested list positions, giver/recipient roles, color and
number answers. Compare gold word targets to within-skill shuffled
word targets with identical training steps and public story rehearsal.

Stage budgets: 120 → 600 → 2,400 distinct training instructions;
250 → 600 → 1,400 updates PER ARM, with public-language loss
rehearsal weights 10% → 20% → 25%.

Use a new final **120-question held-out direct-instruction test**
with new names, words and question structures. The previously tested
50-command and 200-QA evaluations become *development regression
diagnostics*, not fresh generalization evidence.

Report per-skill exact accuracy, both control-arm performance, text
loss and example predictions. Even excellent training performance
without independent transfer is not proof of preschool comprehension.

No user memories, Android observation logs, pretrained third-party
models, or candidate model promotion are permitted.
