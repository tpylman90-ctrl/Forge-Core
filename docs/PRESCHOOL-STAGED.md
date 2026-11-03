# Forge Seed v0.7 — staged QA teaching after failed large curriculum

## Verified diagnosis (November 3, 2026)

The original random-initialized v0.7 model learned basic story structure
after 14,000 public TinyStories pretraining steps. It did not generalize
simple factual questions (19/40 mirrored options, ~chance-level).

An initial verified-label versus shuffled-label experiment used 1,600
original training questions over 1,400 AdamW updates. The verified branch
scored **20/40** on the original held-out benchmark and **20/40** on a
transfer benchmark; shuffled labels scored **21/40** and **21/40**.
These are negative results, NOT evidence of comprehension. The held-out
language loss did not deteriorate (2.6380 frozen to 2.6071 verified).

We next tested a necessary optimization sanity check: can Forge learn the
output format at all when sixteen balanced examples are repeated? The
600-step diagnostic reached **16/16 correct after just 100 steps**,
starting from **8/16**. This demonstrates memorization/optimization
works. It does not establish transfer to new names or sentence phrasings.

## New hypothesis: progressively increase difficulty

Rather than giving a small network hundreds of nearly unique questions
at once, teach in stages:

| Stage | Original questions | Optimizer updates/arm | Public-language rehearsal |
| --- | ---: | ---: | ---: |
| Master answer format | 16 | 200 | 0% |
| Controlled variations | 64 | 400 | 10% |
| More combinations | 256 | 800 | 20% |
| Expanded curriculum | 512 | 1,200 | 25% |

Both branches start from identical pretraining weights. The **verified**
arm receives the rule-correct A/B answer; the **shuffled** control
receives a within-skill class-balanced wrong mapping. Both see the same
questions in identical order, receive the same public TinyStories
rehearsal batches, and have the same stage boundaries and update budget.

At every stage, score (a) accuracy on current training examples,
(b) the fixed 40-choice original QA test, (c) a different 40-choice
transfer test with unseen people and changed wording, (d) held-out
general-language next-token loss, and (e) how many underlying questions
are correctly answered under both A/B mirror positions.

## Capability promotion policy

No claim of preschool comprehension if performance only improves on
training cases. The model must demonstrate transfer (at least 32/40 on
both tests, with at least 12 correctly solved mirrored pairs per test),
retain held-out language loss within 5% of the frozen checkpoint, and
be tested on open-ended instructions and independent prompts before
any promotion. These test thresholds are preliminary, not formal
educational certification.

Neither branch is allowed to replace the archived original model unless
subsequent independently reproduced evidence supports promotion. A
perfect memorization score is an engineering diagnostic, not intelligence.

Source and evidence:
- [Scratch pretraining](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41572257363)
- [Preschool baseline](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41573510393)
- [Failed broad QA teaching](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41635128033)
- [Successful sixteen-question memorization check](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41635991947)
