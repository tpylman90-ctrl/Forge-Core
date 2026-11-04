# Forge Copy v6 — Original trainable character-aligned decoder

**Preregistered research experiment; no neural promotion or Android integration.**

## Evidence and design choice

Forge's previous [unbiased v5 experiment](ENGLISH-UNBIASED-COPY-V5-RESULTS.md)
learned some input-sensitive indexed-letter retrieval, but verified
paired instruction achieved only **173/1,016** exact random-source
letters, **3/80** both-twin exact changes, and **zero** successfully
assembled whole strings. A subsequent non-training word/nonword
challenge yielded **166/856** exact letters and zero whole strings.
Increasing repetition of the original BPE answer-next-token task
is no longer an informative first choice.

A [source-attentive pointer-generator family](https://aclanthology.org/P17-1099/)
is an established technique for preserving details from a source
rather than relying on a model to remember exact word spellings.
Our experiment does **not** import or copy pretrained source weights
from any model: it tests a **new, original random-initialized,
character-aligned neural decoder** that could one day be an
accessory to the independent Forge Seed language system.

**Important architectural difference:** this v6 model is *not*
Forge Seed's existing BPE Transformer. It is a standalone scratch
neural component being evaluated for usefulness. The original
TinyStories-trained language checkpoint is authenticated but
**not used as initialization or updated**. If v6 performs well,
a separate integration and retention experiment must follow.

## What the decoder actually does

The source is represented as individual a–z character IDs,
plus a source EOS marker and padding. In each generation step, a
small decoder receives its own previous outputs and trainable
source-character embeddings with sinusoidal position features.

The pointer arm computes learned source-to-output attention
and generates a **vocabulary distribution over the 26 letters
plus a learned EOS**. The Python evaluation harness does **not**
select the correct source index or insert the correct character
after any network prediction. A complete output must be generated
by the decoder from its own preceding symbols, with EOS emitted
by the model. Length is implicit in the source and EOS marker,
not supplied as a desired answer.

This is *not* a deterministic exact-copy algorithm; it can
and likely will make errors. A non-neural exact Python copier
serves only as a 100% oracle reference in unit tests.

An optional separate attention-position supervision term
provides explicit verified labels for which source slot should
be attended at each output step, including EOS. These labels
enter the **training loss only**, never the forward inference
inputs. Because the architecture has source-aligned
inductive bias, even high exact-copy accuracy is not evidence
of broad language understanding.

## Five matched-update comparisons

Every model starts from an *independent identical random seed*
for its neural weights and sees the same original batch multiset
with a fully teacher-forced next-symbol objective.

1. **pointer_alignment** — trainable cross-source attention,
   supervised characters plus 0.35-weighted verified source-slot
   alignment loss.
2. **pointer_token_only** — the same exact architecture and
   examples, but character/EOS loss only, no gold attention labels.
3. **pooled_source** — same embedding/decoder parameterization
   but only global pooled source representation, no slot-wise
   attention (architectural ablation).
4. **pointer_source_hidden** — same supervised attention architecture
   and target answers, but hide ALL source letters in training and
   inference; source length and EOS remain available. This is a
   necessary source-visibility control.
5. **pointer_wrong_targets** — same slot-wise attention but
   wrong source-target mapping, with target strings shuffled
   *within original source length* so target letter frequencies
   and length distributions remain unchanged.

The last two controls receive insufficient/corrupted information
by design; they are not equally informative teacher conditions.

## Data, tests and compute limits

Train 512 novel **3–8-character** source pairs per seed, with all
source characters sampled uniformly from a–z. For each string produce a counterfactual
twin changed at exactly one index. All complete source strings
are unique between all splits, with an independent deterministic
SHA256 family partition and collision assertions.

New tests, never used by the v5 runs:

- **Development:** 80 novel pairs, same 3–8 length range.
- **Unseen short:** 128 novel pairs, 3–8 characters.
- **Longer transfer:** 72 novel pairs, **9–12 characters**, out of
  training range, still entirely lowercase ASCII.
- **Readable-word transfer:** 36 assistant-curated English
  words, each with a deterministic one-letter-edited near-match,
  with no v6 train/dev/test source overlap.

All test answers are reconstructed by a separate exact string
oracle, not generated by the model or supplied as output inputs
at inference. Evaluation reports exact complete outputs,
both counterfactual twins completely correct, both changed letters
correct, whether outputs differ, character edit rate, and actual
generated examples. No teacher forcing during evaluation.

Development-only checkpoint selection prioritizes number of
fully correct twin pairs, then correct whole strings, then
edit distance. Test exams are inspected only **after** selection,
and treated as permanently used after the pilot.

Two original neural initialization/data seeds **13631, 13632**,
**800 optimizer updates per arm per seed**, 8 training examples
per step, 5 arms/seed. The initial pilot is **8,000** bounded
optimizer updates across 10 models, on one serialized CPU GitHub
Actions job, with a hard 45-minute ceiling and no recurrence.
Use 2 CPU threads to avoid oversubscription.

## Safety and attribution

The v6 model can potentially learn genuine whole-string
*source-to-output* generation, but this is a narrow algorithmic
copying task. It does **not** demonstrate word meaning, spelling
rules, subjects, verbs, prose comprehension, independent
conversation, an Android assistant, or human-level reading.

Unlike earlier v2–v5 adaptations, v6 never trains or assesses
the upstream Forge Seed language model's story-language loss.
There is **no question of original-story regression from these
updates** because that network's weights are not modified or
loaded into the trainable decoder. That is an isolation guarantee,
*not* evidence that a merged future system will preserve language
ability. Merging is prohibited without a separately verified
end-to-end language/grammar and new copying benchmark.

All neural weights are new-from-scratch research-only artifacts.
No phone memory, user's personal data, Android APK, signed
keystore, RLVR policy or arbitrary code interpreter is involved.

## Promotion criteria

A successful research signal must be reproducible across both
seeds, outperform pooled, token-only or corrupted/hidden controls
as appropriate, perform on **novel strings** and true near-match
pairs, emit EOS without an externally supplied loop index, and
show meaningful transfer to **longer strings**. If strong only
within 3–8 characters, the next step is length-generalization
research, not immediate deployment.

**This experiment does not authorize changing the canonical
Forge Seed network or the Android UI.**

Source: `src/forge_core/neural/english_learned_pointer_v6.py`.
Tests: `tests/test_english_learned_pointer_v6.py`.
Workflow: `.github/workflows/english-learned-pointer-v6.yml`.
