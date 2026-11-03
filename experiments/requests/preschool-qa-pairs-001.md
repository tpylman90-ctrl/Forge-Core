# Forge Seed v0.7 — paired counterfactual research

The earlier v0.7 staged curriculum achieved 98.83% accuracy on its
512 *training* questions, but only 18/40 (45%) on the distinct transfer
test. The shuffled control reached 19/40 (47.5%), providing no
generalization evidence. Ordinary language loss remained close to baseline
(2.654611 vs 2.637959 frozen). No candidate was promoted.

Authorized one-time, CPU-bounded research run to test a different teaching
objective rather than simply increase optimizer steps:
- Base: original audited 14,000-step Forge Seed v0.7 public checkpoint,
  Actions run 41572257363; original weights/optimizer unmodified.
- Generate verified counterfactual semantic states for four tasks: color,
  recipient, location, count. Changing the story/question changes the
  correct answer. Every semantic state is represented with BOTH swapped
  A/B option orders. Correct answer cannot be predicted from position alone.
- Five stages of 32, 64, 128, 256, 512 *questions*, each retaining the
  previous questions. Respectively train 250, 400, 650, 850, 1100 steps.
- Gold answers vs within-skill balanced shuffled labels, identical prompts,
  optimizer/schedule/compute; public text rehearsal rises from 0% to 25%.
- Evaluate two preexisting held-out A/B benchmarks, including both-mirror
  accuracy, and held-out TinyStories next-token loss after every stage.
- The transfer dataset includes names absent from training; synthetic
  templates remain narrow, so even success needs further validation.
- Experimental research models kept separately. No promotion, Android
  integration, personal memories, pretrained neural weights or observation
  data. Discard hypotheses that fail unseen transfer.
