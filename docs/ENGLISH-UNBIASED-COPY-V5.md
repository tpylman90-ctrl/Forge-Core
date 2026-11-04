# Forge English v5 — genuinely unbiased character copying

**Status: preregistered scratch-language research only. No promotion.**

## The evidence prompting v5

The successful [v4 study](ENGLISH-POSITION-COPY-V4-RESULTS.md)
found a measurable neural source-position copying signal. On two model
seeds, mixed teaching got **214/498** individual letters, against
**67/498** for shuffled letters, and **173/402** correct letters at
positions 2 and later. But **84/96 second-position answers** were
correct, because the synthetic alphabet alternated consonants and
vowels. On positions 3–7, the same models achieved only **89/306**
versus **37/306** in the shuffled control. Separately querying and
externally assembling whole strings succeeded only **3/96** times.

We must not interpret easy vowel-position shortcuts as robust copying.

## New unbiased source and exact oracles

Use **independently sampled a–z characters at every string
position**, 4–8 characters long, with no vowel/consonant parity
pattern. No real words or externally pretrained teacher AI.
Each original source has one **counterfactual twin** with exactly
one changed letter. A family is explicitly assigned by fresh SHA256
partition to train, development or final test. The two twins stay
together in one split, and the train/dev/exam task generator checks
string collisions and exact changes.

For every member of every family, generate a one-based question
for **each** source position. The correct answer is calculated
independently using `source[index-1]`. This is a known letter
retrieval task, not abstract spelling knowledge. Question wording
differs among train, development and final examination.

The principal novel test is a hard counterfactual:

- Source A: `m q p t`, position 2 → correct `q`
- Source B: `m r p t`, position 2 → correct `r`

The model must generate **both** correct letters at the changed
position, not the same guess twice. We record complete
counterfactual-pair success and whether its two predictions differ.
Neither criterion alone is a substitute for full exact string
assembly, which is reported separately.

## Equal-update controlled interventions

Train four independent research copies of the **same authenticated
Forge Seed v0.7 checkpoint**, random-initialized before the original
public TinyStories pretraining:

1. **Paired verified:** complete, correct labelled twin prompts
   are presented together in each four-example gradient minibatch.
   60% of pairs focus their single changed position.
2. **Unpaired verified:** the **exact same individual examples and
   label frequencies**, shuffled into other four-example minibatches
   so twins are not deliberately co-presented. This distinguishes
   paired counterfactual exposure from plain supervised examples.
3. **Permuted labels:** identical paired input questions but target
   letters shuffled **within the requested-position category**,
   preserving answer-frequency marginals while breaking exact source
   correspondence.
4. **Masked source:** identical pairs and valid labels but replaces
   every input letter with `?`, retaining string length and query
   index. This tests how well the network exploits structure or label
   priors **without the actual source letters**.

All arms use 80% target-letter loss and 20% public-story rehearsal,
the same optimizer setting, cloned source weights, batch size,
update count and held-out loss measurement. Supervisory information
differs by design in the null arms; mixed-versus-paired retains an
identical multiset of question–answer examples.

The pilot uses **900 updates/arm/seed**, **two training seeds**
11631 and 11632, one sequential GitHub-hosted CPU runner with a
**45-minute hard job timeout**. One-off only; no unbounded self-training
or automatic repeats.

## Exam and promotion rules

Use **180 train source families/seed**, **18 development families**,
and **40 final source families** shared across seeds. Every family
includes **two source variants** with all positions queried. The
final exam therefore contains several hundred single-letter questions
per model and **40 hard counterfactual pairs**. The same questions
are asked of both independently adapted model copies; sums across
seeds are replicate trials, not entirely different exam items.

Checkpoint selection uses development-only:
hard-pair both-correct count, correct letters at indices 3+, then
overall letter count. The final synthetic exam is used once after
selection. Report raw denominators, per-position results, especially
positions 3–8, and hard-pair both-correct scores by seed/arm.

Secondary measures: individual generated-letter exactness,
external assembly of all separately queried letters, newline
termination, source-blind performance, and held-out public-story
loss on identical fixed **48 × 64-token** windows.

**A perfect exact Python source-position oracle is a reference
checker, not a neural capability or earned performance.** External
assembly is a scaffold that requests one letter at a time, not
autonomous word generation.

Genuine evidence for progress requires better performance than
all matched controls on counterfactual hard pairs and indices
3–8 across both seeds, **without** crossing the provisional
5% public-language loss regression threshold. Even that would
not demonstrate full word production or human English comprehension.
A meaningful independent reading claim would additionally require
human-authored source text and new operation families.

## Isolation

Only `src/forge_core/neural/english_unbiased_v5.py`, its tests,
documents, and its own GitHub Actions workflow may be changed.
Do not touch `android/`, signed APK workflows, the original
pretraining checkpoint, user-device observations, encrypted personal
memories, or the separate RLVR proof research.

No research weights are automatically promoted to Forge Seed.
A negative run must be archived before attempting architecture
changes; if source-index copying still fails, consider an original
pointer-like mechanism explicitly representing source positions
as a separate controlled architecture study.
