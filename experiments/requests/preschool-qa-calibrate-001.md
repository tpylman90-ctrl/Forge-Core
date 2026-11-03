# One-time QA learning-path diagnostic

The first question-answer training run used 1,600 examples and 1,400 updates,
but both generalization scores remained near 50% and training quiz loss
stayed around log(2). Before scaling, run a 16-example repeated-training
sanity control to check whether the model can learn the question/answer
token format AT ALL.

- Source: original scratch Forge Seed v0.7 run 41572257363.
- Sixteen balanced original questions; four skills; two A/B answers.
- 600 bounded AdamW updates, batch 4, learning rate 3e-4.
- Report starting and final accuracy on the SAME examples plus loss
  trajectory. High same-example accuracy means memorization is possible,
  **NOT** generalization or preschool comprehension.
- No personal memory, pretrained weights or Android modifications.
- Preserve original checkpoint. This run does not publish candidate weights.
