# Pointer transfer exam — first untouched research gate

A 73,920-parameter trainable source-span pointer head on the *frozen*
Forge Seed language backbone scored **42/120** on Teacher v1's
internally generated development questions. An untrained head scored
7/120 and a head trained to select wrong source words scored 4/120.
All three shared the same question set. The original language model
remained frozen.

Because the development questions use known task generators, this is
only evidence of *learned source-word selection on that task family*.
It must transfer to different names, objects and sentences to be
more than template memorization.

The next experiment evaluates **360 untouched source-word questions**
from three predetermined random seeds, with different people and
objects from all preceding lesson sources and deliberately different
sentence constructions. Every expected answer is directly present in
the input and verified by independent original fixture rules; no
pretrained teacher supplies gold. No gradient updates occur in the
exam. All candidate source SHA256 hashes are rechecked before scoring.

Even strong results remain synthetic and insufficient to certify
preschool-level comprehension. The pointer cannot invent missing facts
and may fail when a question requires an answer not in the text.

GitHub Actions run:
https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41670001958

The separate local GGUF AI teacher remains opt-in for curriculum advice,
not for inserting unverified answer labels or pretrained model weights.


## Completed result — November 3

The source-pointer training run passed with **42/120 development**
examples correctly copied, compared with 7/120 untrained and 4/120
wrong-source control.

The first 360-case exam run safely failed before a valid report due to
two duplicate source questions across seeds. A neutral session
context fixed those duplicates, retaining correct answers and the
previously frozen weights. Both retry CI and examination passed.

[Corrected cross-template exam 41671188473](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41671188473):
**41/360 (11.39%)** verified pointer versus **12/360 (3.33%)**
wrong-source control. Per seed: 9, 17 and 15 correct of 120, versus
6, 4 and 2 control. Despite the advantage over control, this is
poor transfer from the 42/120 development result. **No promotion.**

Interpretation: a trainable source-span selector does help within a
familiar generator family, but role binding and understanding of
unfamiliar sentence constructions are not solved. Continue with
a richer syntax/role mechanism and independent natural-language exams.
