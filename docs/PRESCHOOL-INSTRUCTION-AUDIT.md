# One-step instruction gate for Forge v0.7

This independent *research-only* audit checks whether a trained
Forge Seed can respond to instructions, not merely continue stories
or memorize a small template-based question set.

Fifty original short commands cover five categories: repeat a word,
select the requested item from a list, extract an owned object's color,
extract a quantity, and identify who received an object.

The questions are not added to any compositional training arm, and
candidate artifacts are only read after the training run has completed.
The same examples are scored on the original frozen checkpoint and
on the verified-answer and shuffled-answer models; models/decoder
must share identical source-checkpoint and tokenizer provenance.

Every question expects a direct short answer, not a multiple-choice
letter. Greedy output is compared by normalized exact match; examples
of wrong responses and scores for each skill are saved.

Scoring 50 internally authored synthetic instructions is useful
evidence but not preschool certification. Stronger claims require
larger untouched naturally phrased prompts, multiple seeds and
human review. No trained candidate is installed in Android or
automatically promoted.

The workflow listens for successful completion of the compositional
Forge Seed experiment and can also be dispatched manually with
a verified candidate GitHub Actions run ID.
