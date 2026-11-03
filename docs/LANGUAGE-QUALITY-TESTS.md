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

## Verified Run 005 findings

Comparison: https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41558718634

- Both checkpoints used the exact same 800/200 TinyStories source hashes,
  model architecture, batch settings and sampling seeds.
- Step 8,000 produced more ordinary story phrases, but invented words,
  unusual agreement and disconnected clauses persisted.
- The sentence-level and dialogue-quality gate **has not been passed**.
- A unit test detected that the first inspection implementation altered
  global RNG state during neural initialization; this was corrected in
  the follow-up commit and independently retested.
- The test provides sample evidence, not an objective fluency score.
