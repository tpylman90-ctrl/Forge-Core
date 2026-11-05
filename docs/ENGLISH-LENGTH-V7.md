# Forge Copy v7 — length transfer pilot

## Verified v6 starting evidence

Successful v6 run 41932114620: new random-initialized source-attentive
character decoder copied **505/512** novel 3–8-character strings
(249/256 and 256/256), but only **26/288** 9–12-character strings
(8/144 and 18/144). It copied **143/144** curated short words and
near-matches. Pooled-source, hidden-source and wrong-target controls
copied zero novel short strings. The token-only pointer scored 492/512.
These are narrow source-copying results, not English comprehension.

## Preregistered experiment

Two independent seeds, 13741 and 13742. For each, compare identical
initial neural weights in two arms: short-only 3–8-character training
against a length curriculum with half 3–8 and half 9–12-character
pairs. The first third of curriculum steps trains only short
examples, and the remainder samples mixed lengths. Both use 512
verified original one-letter-change pairs, 500 AdamW updates, batch
8 strings, .003 learning rate, and .35 gold alignment loss.

Fresh deterministic development and exam seeds 20261105, 20261106,
20261107. Source/twin split firewall checked before training.
No exam-driven checkpoint selection. Report exact whole strings,
both counterfactual twins, changed-letter responses, and edit rates.
The original Forge Seed checkpoint is verified but is NOT used to
initialize or train this separate scratch decoder.

Bounded 2,000 updates total, two CPU threads, 35-minute Actions cap.
All weights remain research-only. Stronger longer-copy accuracy
would not establish word meaning, role binding, or preschool ability.
