# Forge Seed — Fixed-prompt qualitative language inspection

Held-out loss and bits per byte measure next-byte prediction, not speech,
understanding or conversational quality. A small experimental model can learn
text frequencies while still generating nonsense.

The `forge_core.neural.qualitative` module runs **five fixed, nonpersonal
prompts**, under identical random seeds, through two checkpoints. It records
raw decoded continuations, replacement characters and printable-text fraction.
It does not automatically declare text grammatical or intelligent.

Use **Actions → Forge Seed fixed-prompt language inspection**, supply an
earlier and later training run for the *same corpus and model configuration*.
The workflow rejects mismatched data hashes and uploads a side-by-side report
for human review. No personal observations or private memories are used.

This benchmark is deliberately modest and cannot establish intelligence,
generalization to arbitrary English or proper dialogue. Later gates require
multiple unseen tasks, blind human ratings, factual consistency and coherent
multi-turn evaluation. It provides an early, repeatable visual signal to
supplement quantitative loss measurements.

## First controlled evaluation request

The one-time Run 005 comparison checks the step-4,000 and step-8,000
checkpoints trained on the identical 800/200 source corpus. The side-by-side
samples remain public research artifacts, with no user memories or other
private inputs. Human review is required before declaring any language gate.
