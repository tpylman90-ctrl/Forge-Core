# Forge English v4 — explicit source-position learning

**Research-only proposal. No trained weights promoted.**

## Evidence that determines this experiment

The [completed character-copy v3 experiment](ENGLISH-CHAR-COPY-V3-RESULTS.md)
found a reproducible but limited advance over shuffled-label and
story-only controls: verified training got 19/192 (staged) or
21/192 (mixed) exact new answers, overwhelmingly from one-letter
tasks. In both seeds **0/48** verified mixed responses accurately
joined 3–6 unseen letters, and only the first character was often
correct.

The next question is whether the ORIGINAL Forge Seed language model
can retrieve a character at **any specified position** of an unfamiliar
source sequence, instead of merely beginning with the correct character.

## Curriculum and independent tests

A new task asks:

```
Letter position exercise
Letters: c a t
Index: 2
[short question]
Answer:
```

Correct target: `a`, generated as a one-character answer.
The lesson objective supervises the original Transformer to emit the
single requested character. It is not offered candidate answers.

Train: 150 distinct 3–7-letter original synthetic strings per seed.
Development: 28 new strings; final: 48 new strings and different
question wording. Strings are assigned to one split using a fresh
SHA256 hash namespace, so no whole source string overlaps.
Every position in each string is taught/tested and checked against
`source[index - 1]`, independent of its stored label.

**External scaffold:** at evaluation, the test harness separately
queries the neural model for each letter, then deterministically
joins its predictions. Report the *per-position* correctness and
the *externally assembled* whole-string correctness, explicitly
labelled as assembled, not unaided word generation. A rule-only
reference copier is 100%; it must never be counted as neural
achievement. The model does not learn to decide the string length
or where the loop should stop.

## Research controls

Five conditions, every model initialized from exactly the original
audited Forge Seed v0.7 checkpoint/tokenizer:

1. **Position-staged**: first 1–2 character positions, then 1–3,
   then 1–5, then all 1–7 positions;
2. **Position-mixed**: the exact same sample-batch multiset shuffled
   across the optimizer updates;
3. **Shuffled labels**: same staged batches, same position prompts,
   but target letters shuffled within source position;
4. **First-position-only**: same number of updates, but teaches only
   the first letter of each matched training string (a stronger
   shortcut baseline than always guessing);
5. **Public-story-only**: no letters taught, just the authenticated
   TinyStories corpus with the same step budget.

Correct-label, shuffled and first-only conditions train with 80%
explicit character supervision and 20% unchanged public-story
rehearsal. Story-only gets only original text loss. Source weights
are copied and frozen in storage; updates never overwrite them.

Two training schedule seeds, 10331 and 10332; **110 updates per
four position phases = 440 updates per arm per seed**. All five
conditions run sequentially on one hosted CPU GitHub runner with
a 45-minute total job timeout. No automatic recurring training.

Development checkpoint selection uses total letter correctness,
then correctness at positions 2+, and finally externally assembled
whole words. The final synthetic exam is reserved until afterward.
Track exact letter correctness, later-position accuracy,
assembled-string accuracy, newline termination, and generated
examples, separately from the upstream validation language loss
(identical 48 fixed 64-token public-story windows).

## Limits and next gate

This task deliberately removes the complex burden of autoregressively
generating a long word. Even a near-perfect outcome would only show
a **position-query skill with an external assembly mechanism**,
not natural standalone language production. Novel letter
combinations do not constitute new grammatical or semantic concepts.

Success should be independently reproduced across both seeds,
beat first-only and shuffled controls on non-first positions,
and retain prior public-text performance with <=5% next-token-loss
regression. Only then should a later bridging experiment teach the
model to produce an entire character sequence **without indexed
external questions**. That later test needs independently authored
new strings/words and longer positions.

No private user memory, Android build, autonomous action integration,
downloaded pretrained teacher, or checkpoint promotion is used.
