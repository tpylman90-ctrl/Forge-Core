# Linguistic Why Lab Run 003 — Expanded action-role research

The first two measured A/B experiments improved grammar and negation but
left semantic subject/object roles at chance (10/20). Public-language
rehearsal reduced adverse next-byte language loss substantially.

Authorized one-time public GitHub Actions experiment, not a schedule.

- Original source: 8,000-step scratch Forge Seed checkpoint Run 005.
- Same initial weights for verified-label and shuffled-label controls,
  plus frozen reference. Only the gold versus shuffled labels differ.
- Expand deterministic action-role curriculum across 6 original training
  nouns x 5 original training verbs x distinct actor/object assignments,
  with both "who acts" and "who receives" tasks.
- Approximately 330 total training exercises and 330 held-out evaluations
  with **different** eval nouns AND verbs; grammatical templates still
  overlap, so this is NOT general language comprehension evidence.
- Stratified minibatches: about 50% roles, 25% agreement, 25% negation.
- 240 optimizer steps, batch 2, seed 1234, matched between arms.
- Same public-language rehearsal loss weight 0.4 in both arms.
- Evaluate per-skill held-out A/B accuracy and regression on public
  TinyStories validation data.
- Both candidate weights are RESEARCH ONLY, not promoted, not Android.
- Do not upload personal data or privately learned memories.
