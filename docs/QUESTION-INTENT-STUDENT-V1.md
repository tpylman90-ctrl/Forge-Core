# Forge question-intent student study v1

## Research question

Does a trainable question head over Forge's frozen, scratch-trained language encoder generalize from hand-authored question families to unseen wording and vocabulary? This study isolates the question signal only; there is no story or event context.

## Frozen source and controls

- Use the independent 50 MiB / 5 MiB TinyStories model if its run completes successfully. Verify the checkpoint, pinned source manifest and tokenizer hashes before feature extraction.
- Freeze every language-model parameter. Extract the last-token contextual representation once for each question.
- Train the same randomly initialized projection and 3-way classifier under three conditions: plain cross-entropy, equal-weight cross-entropy plus supervised contrastive loss, and a balanced shuffled-label control.
- Use three initialization seeds and identical 600-update budgets. Select each checkpoint on development accuracy only.
- Score the 90-question final split after all checkpoint selection. Report aggregate, per-class, confusion matrix, every seed, and the 1/3 chance reference.
- Save research-only head weights and verify that the source checkpoint file hash is unchanged.

## Decision gate

No role head is merged into Forge Seed or Android. Consider the method informative only if the correct-label arms beat the shuffled control on the unseen final wording across all seeds and show balanced giver/recipient/other recall. A high score on these internal synthetic questions is not a claim of reading comprehension; a separate story-context challenge with new syntax and independent labels remains mandatory.
