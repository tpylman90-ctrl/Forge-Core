# Forge Seed v0.7 — retry compositional short-answer research

Initial run 41649448511 stopped safely before training when the older
direct-answer evaluator rejected a new fifth category, "giver" (KeyError).
No model weights were adapted or overwritten.

This update makes the evaluator category-agnostic and adds an explicit
regression test for scoring giver answers. All research constraints
remain unchanged: original archived source, 3,200 training items,
five skills, separate 50-case progress set and untouched 200-case
final challenge, correct-vs-shuffled matched training, retention
of public-language rehearsal, and no experimental model promotion.
