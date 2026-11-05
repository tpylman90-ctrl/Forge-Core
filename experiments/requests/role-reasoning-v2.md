# One-time three-seed structured role-reasoning diagnostic

Run the preregistered role-reasoning v2 study against the verified scratch
checkpoint from the 50 MiB / 28,000-step public TinyStories run 42085634121.

## Question

Does an explicit, deterministically verified giver/recipient explanation
improve role binding when Forge must answer with either a person's name or
a candidate letter, and can candidate-letter scoring separate role selection
from spelling an unfamiliar name?

## Controlled design

- Three independent training seeds: 101, 202 and 303. Each seed generates
  its own 3,200-row counterfactual training set.
- Preserve the same frozen source checkpoint, tokenizer and model architecture
  across all arms. No personal data and no external AI teacher.
- Four equal-update arms per seed: structured role explanation plus exact
  name, structured role explanation plus candidate letter, within-role
  shuffled structured labels, and a text-only continuation control.
- Use the same stage sizes, optimizer schedule and number of updates. The
  text-only arm performs two language batches per update to approximate the
  supervised arms' compute. Record the compute proxy and language retention.
- Evaluate once after training on a shared, untouched 240-row final set with
  disjoint names, objects and story/question templates. Half uses familiar
  transfer verbs in new constructions; half uses verbs absent from training.
- Report exact name and candidate-letter accuracy separately, role-trace
  accuracy, paired counterfactual accuracy, per-slice results, held-out text
  loss and samples. Keep all candidate weights research-only.

## Stop / interpretation rules

Stop if any integrity check fails. Do not inspect final metrics for checkpoint
selection, do not promote weights, and do not integrate candidates into Android.
This is still an internally authored synthetic diagnostic; even replicated
improvement does not establish natural-language comprehension. An externally
sourced, licensed role-labeled evaluation remains necessary before broader
claims or promotion.
