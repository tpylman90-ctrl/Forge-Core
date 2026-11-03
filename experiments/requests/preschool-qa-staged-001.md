# Forge v0.7 staged learning experiment

The initial v0.7 source model successfully learned coherent story-like
phrasing but scored 19/40 on mirrored simple-story QA. Training 1,600
examples with 1,400 updates failed to improve transfer. A direct calibration
then reached 16/16 on the SAME repeated questions after 100 updates.

Authorized one-time staged experiment:
- Frozen original checkpoint: GitHub Actions run 41572257363.
- Gold-label adaptation and a shuffled-label control, same original
  weights, source tokenizer, source dataset and matched optimization.
- Stage 1: 16 original balanced questions × 200 steps (learn answer format).
- Stage 2: 64 questions × 400 steps, 10% shared public text rehearsal.
- Stage 3: 256 questions × 800 steps, 20% public rehearsal.
- Stage 4: 512 questions × 1200 steps, 25% public rehearsal.
- Same names and original gold correctness for repeated early questions,
  random class-balanced shuffled labels for the control.
- Evaluate both existing 40-choice tests, explicitly score paired-mirror
  consistency, and held-out general-language loss at every stage.
- Neither model is promoted or deployed. Personal memories, observer data
  and any pretrained neural weights remain excluded.
- This is a research hypothesis about curriculum scaffolding, not a claim
  that model reasoning, preschool fluency or instruction following exists.
