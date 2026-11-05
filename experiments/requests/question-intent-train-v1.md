# One-time question-intent student study

Run the preregistered three-seed comparison using the completed wider-corpus v0.7 checkpoint from successful run 42085634121.

- Verify the checkpoint, tokenizer, pinned source manifest, license and hashes.
- Keep the language encoder frozen.
- Use 600 updates per arm and seed for plain cross-entropy, paired contrastive plus cross-entropy, and balanced shuffled labels.
- Select on the 60-question development split only. Score the 90-question final split once.
- Report each seed, class, confusion matrix, balanced accuracy, benchmark hashes and unchanged checkpoint SHA.
- Save head-only research artifacts. No promotion or Android integration.
