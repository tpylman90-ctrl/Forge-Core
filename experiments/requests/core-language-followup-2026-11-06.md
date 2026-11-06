# Approved cautious follow-up from validation-best 99k checkpoint

Continue actual Forge language training from the validation-best 99,000-step
candidate in sampler-seed-73 artifact from run 42289055100.

- Parent checkpoint SHA-256:
  `b1ba557fc08b2a93c49857499f8d5a13c76eb7a5eb78a94254018876350631ad`.
  This parent was selected by validation loss, not by test loss.
- Resume Forge's own weights, AdamW moments, Torch RNG, and sample RNG.
  Preserve the exact original tokenizer and training/validation text.
- Continue uniform TinyStories training only, at learning rate 0.00005, for
  at most 2,000 updates. Evaluate the same 2,048-window validation split every
  500 updates and stop after two checks without at least 0.0001 improvement.
- Create a new independent test slice from the pinned TinyStories training
  stream, 4 MiB after the previously evaluated range. Remove exact duplicates
  of train and validation stories, record byte ranges and hashes, and use it
  only for final parent/candidate comparison.
- Preserve the 99k parent, selected best-validation candidate, final attempted
  checkpoint, optimizer and RNG state, data, hashes, metrics, and capability
  results as Actions artifacts and an unpublished draft release.
- Mark all outputs research-only. Do not promote or integrate any checkpoint
  into Android.

The replicated 95k-to-99k lower-rate runs improved fixed held-out text
prediction across three sample orders, while toy capability checks remained
at chance. This follow-up tests whether a smaller continued step improves a
new data range; it must not be described as proof of comprehension or reasoning.
