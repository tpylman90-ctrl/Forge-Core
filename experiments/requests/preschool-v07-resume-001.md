# Forge Seed v0.7 — bounded language continuation

One-time research run for the Friday early-language checkpoint.

- Resume the unchanged, audited original scratch-trained checkpoint from successful run 41572257363 (14,000 prior updates).
- Run one additional bounded 14,000-step block with the same pinned public TinyStories source, train-only original BPE tokenizer, optimizer, batch, sequence length, and seed defined in the existing workflow.
- Keep this as a separate research artifact. Do not overwrite the original checkpoint or merge any QA/teacher heads.
- Use the existing automatic capability audit after training; report held-out language loss, fixed-prompt generations, and each basic skill separately.
- No automatic promotion. A lower loss alone does not pass the Friday language, comprehension, or instruction rubric.
