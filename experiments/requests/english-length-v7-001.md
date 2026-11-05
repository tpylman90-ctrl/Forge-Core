# One-time Forge v7 length-generalization pilot

Run one bounded CPU experiment comparing a short-only (3–8
characters) scratch neural pointer copier with a staged mixed-length
(3–12) copier. Two independently initialized seeds, 500 updates
per arm, 512 original changed-one-letter pairs per arm, 8 strings
per update. Matched optimizer, architecture, source verification,
new dev/exam generation, and no model promotion.

Latest v6 achieved 505/512 novel short exact strings across seeds
but only 26/288 unseen longer 9–12-character strings. Test whether
long-string exposure improves the most important copying deficiency.

Generate deterministic held-out short and long exams with no
train/twin overlap. Compare exact whole strings, changed pairs,
and edit rates. Do not select checkpoints using exam scores.
Original Forge Seed weights and Android observer remain untouched.
No personal data, pretrained weights, or preschool claim.
