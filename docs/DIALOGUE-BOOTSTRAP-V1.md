# Pure dialogue bootstrap v1

## Aim

Get Forge Seed closer to a first limited text conversation while preserving
its scratch-trained language lineage. This pilot teaches four narrow
behaviors: recall a fact from the current conversation, accept a correction,
ask for a missing detail, and acknowledge when the user never supplied a
requested fact.

## Data and training

- Source parent: Forge Seed v0.7, run 42552572467, step 107250, seed 43,
  checkpoint SHA-256 `848553a81549c1269c360c5d07c57adde92a7816a0bc19b61ecc8028c93fee99`.
- Original BPE tokenizer and weights are verified before training.
- Every training response comes from a deterministic, reviewed template and
  fictional slots. No external model generates answers; no user chat, private
  memory, or Android data enters the run.
- Correct-target and within-intent shuffled-target arms begin from identical
  copies of the same parent and receive matched CPU updates.
- Development and final exam prompts use disjoint wording and vocabulary.
  The final exam is scored once after the fixed update budget.
- Compare conditional response ranking against the frozen parent and compare
  held-out TinyStories loss for language retention.

The pilot passes only if the correct-target arm beats both the frozen parent
and the shuffled-target control on macro accuracy across all four intents, while
its TinyStories validation loss stays within `+0.05` of the source model.

## Limits and promotion rule

The exam asks the model to rank one rule-generated response against one
intent-matched distractor. This measures a narrow interaction skill, not open-
ended conversation. The synthetic test is an internal pilot and remains
research-only. Do not promote or merge either candidate into the canonical
checkpoint or Android. A later gate still requires fresh wording, free-form
generation review, multiple seeds, and interaction with the user.

To try the raw candidate after downloading the workflow artifact:

```bash
python -m forge_core.neural.dialogue_chat \
  --candidate results/correct.pt \
  --tokenizer results/tokenizer.json
```

The command runs locally, does not call a hosted model, and does not read or
write the encrypted personal-memory vault. It prints the candidate's raw
generation and may produce incoherent text.
