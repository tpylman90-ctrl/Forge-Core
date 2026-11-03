# Forge Seed v0.7 — Pairing answers with meaning changes

**Status:** verified implementation; research result pending.

## Why we're changing direction

The previous staged-learning run
https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41636853840
successfully completed 2,600 updates per arm across 16→64→256→512
questions. It memorized 98.83% of the 512 training questions but failed
to transfer: **21/40** on the original held-out test, **18/40** on the
second test with unfamiliar names/phrasing. The label-shuffled control
scored 23/40 and 19/40 respectively. Both-mirror accuracy was only
4/20 original cases and 4/20 transfer cases in the verified arm.

Language loss (64 held-out windows) was 2.654611 in the verified arm
versus 2.637959 in the frozen original (about +0.63% degradation).
**Conclusion: no validated comprehension, no promotion.**

## New hypothesis

Instead of teaching one random version of each question, explicitly
require correct behavior across two dimensions:

- **Choice reversal:** same story and question, but A/B options swapped.
  The correct label must reverse.
- **Meaning reversal:** change who has the color, who receives the toy,
  where something is placed or whose count is requested. The correct
  underlying answer must change even when both options remain available.

Every generated case yields four related tasks: 2 contrasting semantic
states × 2 choice orders. Synthetic labels are calculated from an
explicit fact relation; Forge never validates its own generated labels.
The first questions are retained as later stages add more cases.

## Controlled experiment

Staged question counts: **32 → 64 → 128 → 256 → 512**.
Optimizer budgets: **250 → 400 → 650 → 850 → 1,100** per arm,
total 3,250 steps. Both independently adapted branches start from
the archived original v0.7 pretrained-from-scratch model and receive
identical questions, minibatch order, LR, AdamW initial state, and
public TinyStories rehearsal batches. Only the gold versus shuffled
target answer differs. A third frozen reference measures baseline.

The two fixed legacy and transfer tests each contain 20 underlying
questions shown in both A/B arrangements. Record per-skill accuracy,
number of underlying items with **both mirrors correct**, training
accuracy, and regression in held-out language token loss. The same
underlying task is not counted as two independent results.

## Decision rule

This method is a promising **hypothesis**, not an established solution.
The candidate is not eligible even for further independent validation
unless it scores at least 32/40 and 12/20 both-mirror-correct on BOTH
fixed evaluations, beats the shuffled control on transfer, and preserves
public-language held-out loss within 5% of the frozen baseline.

Even passing those filters cannot establish preschool-level
understanding: we still need an unseen external benchmark, multiple
training seeds, free-form one-step instructions, and human review.
No trial model automatically replaces a confirmed checkpoint.
