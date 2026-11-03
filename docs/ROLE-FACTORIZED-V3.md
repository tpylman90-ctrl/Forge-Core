# Forge Seed — v3 question intent and event-role factorization

## Reason for the experiment

The 36-minute v2 training succeeded technically, but **both** questions
about the same event were correct in only **95/540** unseen synthetic
pairs (17.6%). The three seeds varied and one shuffled-label control
outscored its verified head. The question-ablated arm had **0/540**
correct pairs, but the other heads still often answered the same name
for opposing role questions.

Instead of extending the v2 optimization horizon, v3 asks whether
two separately supervised abilities learn more reliably:

1. Read a sentence and identify which **detected name** was the giver.
2. Read the **question text** and classify whether it asks for the
   giver or the recipient.

At inference, combine the two predicted distributions. **No gold role
or answer is supplied to the model's forward pass.**

## Architecture and supervision

The original Forge Seed v0.7 language Transformer remains frozen.
The person-candidate detector is a source-only capitalization heuristic,
not a learned general NER model. Candidate embeddings are derived from
the frozen language backbone without including the question.

The new random-initialized head comprises:
- A story-only scorer for the two detected people.
- A separate trainable hashed character-trigram question encoder,
  initialized randomly and updated from **explicit question-intent
  supervision**.
- A probabilistic composition rule combining the two predictions into
  the final candidate answer.
- Independent event-giver and question-intent cross-entropy terms,
  plus a smaller composed-answer term.

**Known inductive bias:** This architecture explicitly separates
people's event roles from the role requested by a question. It is a
research scaffold, not evidence that Forge discovered semantic roles
without any supervision.

## Comparisons

- Factorized head trained with verified event and question labels.
- Identical factorized head, initialization and training schedule,
  but the giver's identity is randomly corrupted by entire paired
  event groups (truth-free story control). Question labels stay valid.
- Question-ablated factorized head: receives a constant question
  embedding; it cannot distinguish queries at inference.
- v2 monolithic, question-conditioned head trained on the **same
  synthetic examples and optimizer rounds**, but using v2's original
  paired contrastive objective. It is not a matched-parameter baseline.

The model learns from **1600 augmented public synthetic questions**
derived from 320 source examples and five manually specified
training-only question paraphrases. Development has **80** separate
examples. A new final **288-question** synthetic examination uses
different names, objects, two fixed seeds (12311, 12312), three
new grammar families (courtyard, observatory, riverside) and two
held-out question wordings per role. The previous v2 final exam
is NEVER reused as the untouched exam for this new study.

Gold answers are reconstructed by independent anchored parsers;
exact duplicate prompts and vocabulary reuse are rejected. All
curriculum material is original synthetic text. This remains
**different from human-authored natural-language comprehension**.

## Measurements and promotion gates

Main outcome is both-correct giver/recipient pairs, not isolated
exact-match accuracy. Also report entity-role accuracy separately
from question-intent accuracy and paired question sensitivity.
A two-choice model can get 50% exact accuracy without learning
anything about the question.

Hold all four arms to the same update schedule, check two independent
random initialization seeds, and select the best weights by
**development data only**. The new final exam is scored once at the
end of each training run.

No experimental head is automatically promoted. We require an
advantage over controls across multiple seeds, above a defensible
two-person baseline, verified external sentences, and language
retention before attaching a new head to canonical Forge Seed.

## Resource plan

A **single** standard hosted GitHub CPU runner, two seeds, at most
**6 minutes of v3 experiment time** by default, and a 20-minute
job timeout to include installation and checkpoint preservation.
This pilot tests architecture and signal quality before another
36-minute run. It saves research-only development curves,
optimizer state, random state and metrics as temporary Actions
artifacts. No jobs run continuously; there is no inferred GitHub
daily compute quota.

Run workflow: .github/workflows/role-factorized-v3.yml.
Authorized request: experiments/requests/role-factorized-v3-001.md.

The original Forge Core language model was trained from random
initialization. Neither pretrained language-teacher weights nor any
private personal memories enter this experiment.
