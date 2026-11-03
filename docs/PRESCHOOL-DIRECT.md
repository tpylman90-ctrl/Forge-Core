# Forge v0.7 — Answer in words, not multiple-choice letters

**Status:** unverified experimental method; no capable assistant is claimed.

## Why

The original Forge v0.7 scratch-trained model has begun generating readable
but repetitive TinyStories-like prose. The first QA curricula trained the
model to output A or B. Even after explicit counterfactuals and mirrored
answer locations, its best held-out transfer result was 23/40, only one
answer better than the shuffled-label control, and too weak to establish
meaningful comprehension.

We now test a different learning target: construct an answer such as
" Ada.", " blue.", or " under." directly, using the same from-scratch
original BPE vocabulary and original model weights.

## Supervision and comparison

1. Build original, explicit fact/object rules for questions about color,
   recipient, place, and count. Each counterfactual changes the right
   answer. Exclude A/B multiple-choice labels and deduplicate both mirrors
   of the same underlying fact.
2. Train on the answer token span only. The model sees the short story and
   question as context; uninformative prefix tokens are masked out of the
   auxiliary loss. It must learn the correct word sequence, not an index.
3. Run two equal-budget arms from identical original weights. One receives
   verified short answer text; the other receives within-skill shuffled
   answers. Add identical public TinyStories rehearsal to both branches.
4. Move gradually from 16 to 64 to 256 synthetic source questions, with
   250, 500 and 1,250 AdamW updates per arm, respectively.
5. Evaluate *greedy direct-answer exact match* against 20 original and 20
   separately phrased held-out cases with names unseen during teaching.
   Record each skill separately, examples of wrong output, and held-out
   next-token story-language loss.

## Scientific limitations

Twenty underlying cases per evaluation set is a small and incomplete test.
Using greedy exact match can fail on valid paraphrases. Synthetic training
templates may reward surface tricks. This result must be repeated with
multiple seeds, an independent open-ended language assessment and larger
natural instruction data before any capability claim. No experimental
model weights replace the original audited Forge checkpoint.

The goal is **language comprehension** and useful responses, not the
ability to select a letter on a quiz. Negative results will be recorded.

## Verified first results

[GitHub Actions run 41643271245](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41643271245)
completed with a separate train-only BPE tokenizer and publicly
attributed TinyStories rehearsal. The model reached **256/256 exact
answers on its practiced training set**. Direct, greedy unseen answers
were only **9/20** on the original set and **6/20** on unfamiliar
transfer (vs **4/20** and **3/20** respectively for shuffled-label
teaching; frozen model 0/20 on both). The general-language loss
changed **2.637959 → 2.661562** in the verified arm. No trial was
promoted. The initial run had been safely stopped before training by
mismatched mirror ID formats; the corrected code validates source
pair contents instead of suffixes. The 20-question holdouts are
small and not a definitive proof of understanding. Wider evaluation
and true one-step instruction tests are the next milestone.
