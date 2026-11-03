# Forge v0.7 — initial direct-answer language apprenticeship

Owner-authorized one-time training run. The earlier 3,250-step paired-choice
experiment achieved only 23/40 (57.5%) on unfamiliar A/B questions versus
22/40 (55%) shuffled-label control; no model was promoted. General language
loss was 2.696069 vs 2.637959 frozen (+2.2%), still within 5% tolerance.

To move beyond multiple-choice strategies, train a tiny teacher-forced
language-output objective. Targets are rule-derived actual names, colors,
counts and location words (e.g. Answer: Ada.), not A/B letters.

- Preserve original 14k-step scratch model run 41572257363 untouched.
- Start two identical, independent neural copies from that checkpoint.
- Verified answer text vs class-balanced within-skill shuffled answer text.
- Source data is synthetic verified pairs, without A/B positions; remove
  duplicate mirrors, retain counterfactual changes to original facts.
- Three stages: 16/64/256 direct questions (32/128/512 paired questions
  before deduplication); 250/500/1250 steps per arm.
- Teacher forcing loss is only on answer tokens, plus increasing identical
  public TinyStories next-token rehearsal (0%, 10%, 20%).
- Measure direct *greedy* short-answer exact match on 20 original and 20
  differently worded transfer questions with unseen people, and held-out
  story-language loss. Sample generated story text.
- Training models are experimental, not yet preschool capable, not approved
  for Android or released to the public. No personal memory, app
  observation, third-party pretrained weights, or unverified model-generated
  labels.
