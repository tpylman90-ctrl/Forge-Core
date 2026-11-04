# Forge Core — Research options after source-position copying

**Purpose:** a *design comparison*, not a claim of implemented
capability or approval to run unbounded compute. Choose the next
engineering route only after the v5 unbiased counterfactual
experiment has produced measured results.

## Current negative/positive evidence

- The original Forge Seed v0.7 Transformer is trained from random
  weights on public TinyStories content.
- English Class v2 open-ended words and sentences scored 0/112 on
  unfamiliar questions in every research arm.
- Letter-copy v3 showed some successful generated first-letter
  answers, but no reliable full-length novel strings.
- Position-copy v4 improved to 214/498 generated individual
  source-index letters with correct mixed supervision versus
  67/498 with permuted labels, but its alternating
  consonant–vowel string generator leaked a large positional prior.
  External full-string assembly remained 3/96.
- v5 fixes that shortcut with independent uniformly random letters,
  single-character-changed counterfactual twins and source-hidden
  controls. This is the main evidence gate.

## Candidate architecture options

### Option 1 — Preserve the original byte/BPE causal Transformer

**Goal:** maintain the same original language model. Represent
source letters as delimited characters, train exact source-index
supervision, then teach whole-string next-character decoding.

**Benefit:** preserves checkpoint lineage and measures how far the
existing network can develop. **Risk:** the tokenizer can merge
characters into multi-character units; absolute positions in the
token sequence need not correspond to source character positions,
and a small model may prefer memorized output patterns.

**Test:** v5's uniform random letters, single-character-changed
twins, and source-blind control. Do not optimize on a previously
inspected exam.

### Option 2 — Add a learned character-aligned pointer/copy mechanism

A pointer-generator is a published neural architecture that can
select tokens from the source in addition to predicting ordinary
vocabulary output. In Forge, an **original randomly initialized
small pointer module** could learn to attend from the requested
character index to a specific source-character slot, with an
explicit learned gate for copying versus generating.

**Benefit:** gives the network an appropriate representation for
preserving exact new symbols. **Risk:** a built-in deterministic
indexed copier could trivially solve the task while hiding
a model that had learned nothing. It must remain a clearly
identified hard-coded oracle baseline, not a neural achievement.

**A valid test:** a learned pointer with gradients, source/target
access audited, compared against the original Transformer with
matched source examples and CPU budget. Independently evaluate
new source lengths and symbols. Report pointer output separately
from unaided language generation.

See See, Liu & Manning, *Get To The Point: Summarization with
Pointer-Generator Networks*, ACL 2017:
https://aclanthology.org/P17-1099/

### Option 3 — Character-level subtask encoder and explicit positional links

An original lightweight character encoder can map each input
symbol to its own position, while the existing Transformer
produces the remainder of the sentence. A learned query-to-source
attention layer can then pick the relevant source position.

**Benefit:** avoids the ambiguity of BPE position alignment.
**Risk:** this becomes a distinct architecture and needs separate
from-scratch training; the character encoder cannot simply
relabel the original model's weights as transferred intelligence.
Any deterministic source indexing assistance must be disclosed.

**Test:** separate baseline model with the same train/test string
sets, compare novel-source letter accuracy and full-string output,
then test whether grammar and TinyStories retention still work.

### Option 4 — Explicit algorithmic scratchpad and variable binding

More structured representations can reduce some algorithmic
generalization burdens, but scratchpads can also harm accuracy
if they make the required program more complicated. Any trace
must consist of **explicit public operations** that can be
independently checked, not a claim to reveal hidden neural
chain-of-thought.

A v6 idea is to teach
`source string → index/position → selected character`
using a small, original, verified step protocol, then test
whether the model can *emit* the next character without an
external indexed request. Avoid leaking the correct character
inside the query itself.

Supporting research:

- Zhou et al., *What Algorithms Can Transformers Learn? A Study in
  Length Generalization*, ICLR 2024:
  https://proceedings.iclr.cc/paper_files/paper/2024/hash/45ed1a72597594c097152ef9cc187762-Abstract-Conference.html
- Ruoss et al., *Randomized Positional Encodings Boost Length
  Generalization of Transformers*, ACL 2023:
  https://aclanthology.org/2023.acl-short.161/
- Jelassi et al., *Repeat After Me: Transformers are Better than
  State Space Models at Copying*, ICML 2024:
  https://proceedings.mlr.press/v235/jelassi24a.html
- Cai et al., *Extrapolation by Association: Length Generalization
  Transfer In Transformers*, NeurIPS 2025:
  https://papers.nips.cc/paper_files/paper/2025/hash/670d983b54992416b8292a171711cdc1-Abstract-Conference.html

## Recommended selection rule

Do not choose a more complicated architecture just because it can
solve our own synthetically generated target through built-in
deterministic indexing. First measure v5 after removing the
alternating-vowel shortcut. If verified teaching is consistently
superior to source-masked and shuffled controls on changed-index
twin pairs, preserve that information and test a small original
source-aligned neural pointer on **new lengths and unseen symbols**.
If v5 learns nothing, diagnose tokenizer/position representation
with small one-step original scratch tasks before committing more
CPU time.

Regardless of route, the success gate includes:
- independently computed exact ground truth;
- position-controlled and changed-one-character counterfactuals;
- generated output without A/B choices;
- externally assembled versus autonomously written words separately;
- untouched human-authored transfer examples before promotion;
- preserved public-text language loss and complete source lineage;
- strict isolation from the Android companion and private user memory.

**No model promotion is authorized by this design note.**
