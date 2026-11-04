# One-time Forge Seed — English Class v1 literacy training pilot

User authorizes the independent-language training track to implement
a school-like curriculum, without touching the Android companion.

- Use the exact audited v0.7 scratch-trained language checkpoint and
  source tokenizer from Actions run **41572257363**.
- Lesson progression: uppercase letters; vowels; joining written word
  parts; noun versus verb; subject versus object; sentence word order.
- Compare ordered lesson blocks with the **same exact lesson batch
  multiset interleaved**, plus a within-skill shuffled-answer null.
- Default **120 updates per level**, six levels, **720 optimizer
  updates per arm**; one standard GitHub-hosted CPU runner with
  50-minute total job ceiling. No auto-repeat schedule.
- Rehearse original authenticated TinyStories text equally in every
  arm and measure retained held-out language loss versus frozen model.
- Strict independent answer reconstruction, balanced A/B positions,
  separate held-out names, nouns, verbs, word parts and question
  phrasings. Select research weights only on development exercises.
- Report each English skill, exam accuracy versus frozen/mixed/shuffled,
  text-language retention and source provenance. Archive
  experiment-only checkpoints; **no automatic promotion**.
- This is text-only multiple-choice grammar groundwork, not proof
  of conversational fluency, spoken phonics or human preschool level.
- Leave `android/`, Android workflows, existing neural proof policies
  and any personal device memory unchanged.

The long-term objective is useful independent English comprehension,
not maximizing a synthetic multiple-choice score.
