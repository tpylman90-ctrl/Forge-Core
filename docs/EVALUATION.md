# Forge Core — Evaluation gates (language first)

## Integrity and privacy

1. Unit tests cover causal masking, finite loss/gradients, checkpoint resumption
   and incompatibility refusal.
2. All training sources require reviewed provenance and license.
3. The schema-v2 document manifest must match train/validation SHA256 and have
   disjoint document IDs and exact content hashes.
4. Data cleaning performs a bounded word-5-gram Jaccard near-duplicate pass.
   This is not a full legal/content or train/test contamination audit.
5. No personal memories, observation logs, secrets or private user data in Git,
   CI logs, public datasets or weights without an additional explicit opt-in.

## Structural-language "why" test track (prototype)

- Diagnostic labels must distinguish **grammaticality** from **semantic
  roles** and **event polarity**. Two sentences may be grammatical yet
  express different events.
- Gold answers come from reviewed fixtures or verified annotations, never
  the candidate model's speculative explanations.
- The current generated lexical split alone is **not** an independent
  generalization result. Add out-of-distribution verbs/constructions,
  independently sourced examples and human review before making claims.
- Compare a text-only control to an auxiliary-task intervention at
  equivalent training compute and multiple seeds. Record structured-label
  accuracy, language-model perplexity and readable-generation regressions.
- A language model's "because" statement is a **testable assertion**, not
  a transparent account of hidden neural causes.

See [Master Blueprint](MASTER-BLUEPRINT.md) and
[Linguistic Why Lab](LINGUISTIC-WHY-LAB.md).

## Measurements for **every** language experiment

- Code SHA, data source/revision/rights, accepted/rejected source documents,
  corpus SHA256, tokenizer, architecture and exact parameter count
- Seed, optimizer, learning rate, context size, batch size, steps, hardware
- Identical deterministic validation windows distributed across the held-out
  data (number of windows and source corpus size included)
- Held-out next-byte loss and bits-per-byte before and after training
- Checkpoint/optimizer/RNG reproducibility and training-resume consistency
- Controlled generation sample and representative failures, when available
- External checkpoint backup/retention status

Loss reduction is not sufficient for language, coherent dialogue or reasoning.
After language emerges, add fixed human-scored fluency, unseen narrative,
question answering, instruction following, multi-turn consistency and
ability to express uncertainty. Never compare against a large commercial
assistant without disclosing the model/data/compute difference.

## Capability promotion

A run may be recorded as a successful *engineering experiment* when tests
pass. It may be promoted as *language learning* only after multiple
independent held-out runs show reproducible gains and samples show meaningful
coherence. Personality, observation and Android integration are later gates.
