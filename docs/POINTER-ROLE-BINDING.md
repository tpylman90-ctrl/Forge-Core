# Forge Seed — independent role-binding diagnosis

**Stage:** Research diagnostic, not new training or model promotion.

## Why this experiment

The existing frozen-backbone neural word pointer achieved **42/120**
on its development tasks but only **41/360** on a new cross-template
synthetic exam (wrong-source control **12/360**). It can sometimes copy
a word, but generalization to unfamiliar sentence constructions is weak.

Before trying to train another head or expose Forge to new data, isolate
two failure modes:

1. **Question conditioning:** Given one sentence, can Forge choose the
   giver when asked *who gave* and the recipient when asked *who received*?
2. **Counterfactual role reversal:** If two people's roles are swapped in
   the sentence but the task remains otherwise identical, does Forge
   change its answer to the new correct person?

## Prespecified design

- Use the **same** audited original scratch-model checkpoint, compositional
  research parent and trained verified/wrong-source pointer heads from
  Actions runs 41572257363, 41649770774, and 41670001958.
- **No updates to any weights**, no gradient computation, no model
  selection, no interaction with user memories or Android.
- Three fixed novel evaluation seeds: **6101, 6102, 6103**.
- Four original semantic structure templates:
  active *handed*, receiver-first *received from*, passive *given to by*,
  and active *sent to*.
- Eight independently sampled people/object situations per template and
  seed, two opposite role assignments, and two different who questions.
  This yields **384 distinct prompts**, **192 paired why/which-person
  questions**, and **192 swapped-entity counterfactual pairs**.
- Enforce disjoint proper names and objects from earlier pointer
  train/development and cross-template exam pools.
- Independently parse each source story using a separately defined
  anchored grammar; reconstruct gold giver/recipient rather than trusting
  the generator's answer strings.
- Score both previously frozen pointer arms and publish breakdowns by
  seed, template, and semantic role. A correct answer is a copied
  **whole source word**, not a substring of a person's name.
- Record **both-correct** semantic-question pairs and **both-correct**
  counterfactual pairs; individual question accuracy cannot establish
  reliable role binding if opposite role questions cannot both be answered.

## Research outcomes and decision gate

This is **diagnostic**, not a new qualifying preschool benchmark. Do not
promote any model regardless of result. Low paired success with occasional
individual correct answers indicates fragile question-conditioned role
binding. If both pairs work but unseen natural-language questions fail,
study training distribution and linguistic diversity before adjusting
model architecture.

The source-word pointer is allowed to use copied strings only. Even 100%
on four programmatically generated templates would establish only
task-specific synthetic performance, **not language comprehension**.
Human-authored and externally sourced examples, diverse independent
syntactic constructions, additional seeds, and language retention tests
are required before making broader claims.

## Run

GitHub Actions: **Forge Seed — frozen pointer role-binding diagnostic**.
The workflow fetches fixed experiment artifacts and verifies SHA256 model
lineage. It saves only JSON results, without storing trained weights.

Local invocation with the exact previously vetted artifact directories:

    python -m forge_core.neural.role_binding_diagnostic \
      --source original --parent parent --pointer pointer \
      --out results/role-binding-diagnostic.json

Provenance: [pointer training](TEACHER-POINTER.md) and
[previous independent transfer exam](TEACHER-POINTER-TRANSFER.md).
