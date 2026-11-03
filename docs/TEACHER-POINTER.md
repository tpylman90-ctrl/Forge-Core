# Forge Teacher v1 — Frozen-backbone source-word pointer

## Why a pointer?

The first Teacher v1 *neural student* study successfully completed but
failed to improve transfer: **22/120** starting development accuracy,
**19/120** after verified training, **14/120** after shuffled targets.
The public TinyStories language loss shifted from **2.691088** to
**2.725105**, inside the 5% regression gate but still worse. The
teacher's subsequent lesson bundle was adapted to these aggregate
development scores and preserved separately. Neither student arm
was promoted.

The main observed weakness remains unfamiliar names and instructions
such as "who gave" versus "who received". More answer-token training has
mostly produced memorization.

A possible solution is an *explicit neural pointer mechanism*. The
frozen Forge Seed Transformer produces contextual hidden states for the
entire input. A new small **randomly initialized** pointer head learns
to select one input word span and copy its original bytes. It can
therefore return an unfamiliar name without having to generate its
BPE tokens from language vocabulary probabilities.

## First experiment

- Load the audited scratch-origin 14,000-step model and a verified
  compositional research branch. Freeze all backbone parameters.
- Load Teacher v1's feedback-selected, checksum-validated 240
  training questions and 120 separate development questions.
- Identify all alphabetic word spans in each prompt, compute their
  representations by pooling the frozen BPE hidden states that overlap
  each span, and use the query hidden state at "Answer:" to rank them.
- Gold spans are the words equal to the trusted rule-derived target.
  Control spans are different words already present in the same input.
- Train two identical randomly initialized pointer heads for 500
  matched updates; compare original untrained, verified, and incorrect-
  source controls on development examples with unseen people/words.
- Keep the source model frozen, so this cannot erase its language
  weights. Save pointer-only research artifacts, never promote them.

This is still an internal synthetic DEVELOPMENT test. Even if the
pointer reliably copies a name, it doesn't demonstrate that the model
understands the transfer verb or the reason for an event. Independent
natural language exam cases and multiple seeds remain necessary.

A local AI teacher is optional and not run in GitHub Actions. All gold
answers are derived from Forge Core's own verified rule generator.
