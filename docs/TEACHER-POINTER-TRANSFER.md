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
