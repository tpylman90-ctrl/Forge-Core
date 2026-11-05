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


## Completed result

The first three-seed study completed in Actions run
[42091686838](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42091686838);
head-only weights and full metrics are in
[the results report](QUESTION-INTENT-STUDENT-V1-RESULTS.md). Plain CE reached
166/270 (61.5%), contrastive+CE 170/270 (63.0%), and shuffled labels 76/270
(28.1%). The supervised heads beat shuffled labels per seed, but contrastive
added only 4 answers out of 270 over plain CE. Giver recall was just 31/90
plain and 33/90 contrastive; both models often called a giver the recipient.
No promotion. Next: explicitly pair counterfactual giver/recipient questions,
then score a new untouched wording split.
