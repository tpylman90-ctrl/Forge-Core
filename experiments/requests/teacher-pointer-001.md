# Forge Teacher v1 — neural word pointer, one-time controlled experiment

Teacher v1 lesson generation and source verification succeeded in GitHub
Actions 41668685911. First student adaptation scored 19/120 development
word answers, vs parent 22/120 and shuffled 14/120; no model promotion.
The next feedback-adjusted lesson bundle was saved but NOT yet trained.

This experiment tests the architectural hypothesis that Forge needs a
differentiable *source-copy pointer*, not more memorized answer words:
- Freeze the existing scratch-origin 2.22M-parameter Forge model
  (verified compositional research branch, original source run 41572257363).
- Derive a small random-initialized neural pointer from frozen hidden
  states; use BPE token offsets to represent words in the input.
- Load the next 240 verified feedback-adjusted Teacher v1 lessons
  with a separate 120-question development set.
- Gold supervision: select the source word exactly equal to the
  original rule-derived answer. Compare an identical initialized
  pointer trained to select an INCORRECT source-present word.
- Exactly 500 steps per arm, equal source questions and indices,
  source model/tokenizer SHA256 verification, no original backbone updates.
- Report direct source-span accuracy overall and by skill, not only
  training accuracy. Do not use this development set for checkpoint
  promotion or claim preschool language.
- Keep every result research-only. No private memory, pretrained teacher
  weights merged into Forge, Android deployments, or external actions.
