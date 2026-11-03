# Forge Seed — neural role-conditioned entity selection (research v1)

**Goal:** Determine whether a separate randomly initialized, question-conditioned
neural head can learn who gives or receives an object, rather than
merely copying arbitrary words from a story. This is a controlled
research experiment, **not** an AI capability claim.

## Motivating evidence

Our previously frozen word-pointer head produced **83/384** correct
answers on the four-template role diagnostic, but only **9/192**
events had both giver/recipient questions correct, and only **11/192**
counterfactual name swaps had both correct. The earlier pointer's
unfiltered 360-case novel-template accuracy was **41/360**.

See [verified prior audit](POINTER-ROLE-RESULTS.md).

## Experimental architecture

- Recover the exact Forge v0.7 **scratch-trained** BPE Transformer
  lineage from Actions runs 41572257363 and 41649770774.
- **Freeze every backbone parameter**. None of the original language
  weights are updated or replaced with another AI's pretrained weights.
- Detect two proper-name candidates from the story using a
  capitalization-based heuristic **without reading the correct answer**.
  This deliberately limited detector is *not* a trained general NER model.
- For each detected name, use its original BPE hidden embedding, a
  nearby contextual embedding, and simple non-gold position/preposition
  features.
- The natural-language question is encoded with the same frozen model.
  A new trainable neural scorer ranks both candidate people **conditioned
  on that question representation**. The head's forward pass never
  receives the expected answer or a verified giver/recipient label.
- Compare the trained head to an equally initialized **shuffled-answer**
  head using the same update schedule and data, the untrained head,
  and the historical pointer **restricted to the same two candidates**
  (an informational comparison, not matched-compute training).

## Preregistered data separation

| Set | Cases | Names/objects | Syntax | Seeds |
|---|---:|---|---|---|
| Train | 320 | Train-only | offer, accept-from, lend, borrow-from | 7711, 7712 |
| Development | 80 | Dev-only | deliver-to, collect-from | 8811 |
| Final synthetic exam | 288 | Exam-only | entrust-to, courtesy-of | 9911, 9912, 9913 |

For each source event, both role questions and both reversed-person
counterfactuals are present. Different question wording is reserved
for development and final synthetic exam. Sentence gold roles are
reconstructed by independent anchored grammar parsers; their string
answers are not treated as a source of detected candidates.

The **label-shuffle control operates within question role, but across
templates**, preserving answer-position balance while destroying the
sentence-form/answer association. Shuffling within each template
would be a broken control because each template fixes name order.
Tests explicitly reject an unchanged shuffled target sequence.

## Training and evaluation gates

Two heads receive identical random initialization and CPU optimizer
schedules. Default **350 updates per arm**, batch 8, AdamW 0.001.
No private memories, Android events, additional third-party pretrained
weights, or online teacher prompts are used.

Publish full breakdowns by role, syntax template and seed. The main
quality gates are **both-correct giver/recipient pairs** and
**both-correct changed-entity counterfactual pairs**. Individual
exact-name accuracy can be misleading. Two-name classification has a
nominal 50% random-choice baseline; the historical wrong-source pointer
had a very different candidate space, so its low result must not be
presented as a chance baseline.

The historical pointer restricted to two detected names is an
informational baseline, not a fair equivalent training arm.

Even a positive result on these small synthetic grammars requires
replication across random seeds and **independently authored,
natural-language questions** before model promotion. If the verified
head fails to outperform shuffled/position baselines convincingly on
the new syntax, we should prioritize richer general-language data,
representation quality and evaluation rather than blindly scaling the
same synthetic examples.

## Workflow and artifact

GitHub Actions: **Forge Seed — question-conditioned role learning pilot**.

Reuses only source-verified original data/checkpoints. Saves
`role-conditioned-report.json` and two new **experimental-only heads**.
These are not promoted to Forge Core, Android or user memory.

[Full original role diagnostic](POINTER-ROLE-RESULTS.md) ·
[Master Blueprint](MASTER-BLUEPRINT.md)
