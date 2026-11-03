# Forge Seed v0.7 — first balanced question-answer teaching study

A one-time, owner-authorized, public-data-only development experiment to
improve the actual skill missing from the current model: short-story
question answering. Preschool capability is NOT presumed.

- Original model: published-to-Actions research artifact from scratch run
  41572257363, 2,219,136 trainable parameters, 14,000 pretraining steps.
- Two independent adaptations begin from identical Forge Seed weights:
  verified synthetic question labels versus within-skill balanced shuffled
  labels. Frozen original is scored too.
- Four skills: color, recipient, spatial location, number/count.
- 1600 original deterministic training questions on names never used
  in the two held-out 40-choice test sets; A/B answer positions balanced.
- All arms use fixed identical prompts, minibatch indices, CPU budgets,
  optimizer and exactly the same public TinyStories next-token rehearsal.
- 1,400 optimizer steps per trained arm, batch 4; 25% public language
  rehearsal and 75% verified/shuffled QA, AdamW lr 8e-5, seed 913.
- Evaluate: preexisting mirrored preschool QA suite; a separate
  40-choice transfer suite with different names and story phrasings;
  held-out general-language loss; reproducible story samples.
- Neither experimental checkpoint is promoted, integrated into Android,
  or used for personal learning. No private memory, observation data
  or pretrained model weights are present.
- Honest result: if scores fail or language regresses materially, do not
  claim preschool achievement; redesign curriculum or improve language.
