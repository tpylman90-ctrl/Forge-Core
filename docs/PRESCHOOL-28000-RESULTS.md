# Forge Seed v0.7 — 28,000-step continuation results

**Completed:** November 5, 2026  
**Research run:** [42081671731](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42081671731)  
**CI:** [42081671274](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42081671274)  
**Artifact:** [forge-seed-v07-research](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42081671731/artifacts/13441422924) (expires December 5, 2026)

## Design and lineage

This bounded run resumed the preserved 14,000-step, randomly initialized Forge Seed v0.7 checkpoint from run 41572257363 for another 14,000 steps. It used the same pinned TinyStories revision, the original scratch-trained BPE tokenizer, seed 42, batch size 4, sequence length 128, and public-only data. No teacher labels, synthetic QA adaptation, personal data, Android memory or model fusion were used. The original checkpoint remains preserved; this run produced a separate research artifact.

Total training reached **28,000 steps** in about 14 minutes 20 seconds of model training. The fixed validation token loss changed from **2.637959** at the 14,000-step starting checkpoint to **2.431283** after continuation (about 7.8% lower).

## Friday capability probe

| Measure | Result | Provisional criterion |
| --- | ---: | ---: |
| Mirrored forced-choice QA | **19/40 (47.5%)** | At least 32/40 mirrored answers, equivalent to 16/20 underlying items |
| Both answer-order mirrors correct | **0/20 pairs** | Counterbalanced evidence required |
| Color | 5/10 |  |
| Count | 5/10 |  |
| Location | 5/10 |  |
| Recipient | 4/10 |  |
| Sentence continuation | No formal human-scored 10-item evaluation | 8/10 readable, complete, grammatical and relevant |
| Two-sentence story | No formal human-scored 5-item evaluation | 3/5 coherent and linked stories |
| Single-step instruction | Not measured by this automated audit | 8/10 unseen instructions followed |

This A/B score is not a comprehension certificate. Its own audit notes that first-token forced-choice scoring remains vulnerable to prompt-template and answer-position shortcuts. The zero fully correct mirrored pairs is a clear sign that the 19 single-case correct results do not yet establish stable relation tracking.

## Generated text review

The six fixed-seed continuations were inspected qualitatively in the workflow output. They contain some grammatical clauses and recognizable story vocabulary, but the sample set shows repeated phrases and characters, inconsistent events, awkward word choices, and story drift. This was an informal review of six prompts, not a blind human-rated sample and not a formal score against the sentence/story thresholds.

## Decision

**No preschool-like readiness pass and no checkpoint promotion.** Broad next-token training improved held-out story loss and some surface fluency, while the fresh QA probe remained at chance and generation remains inconsistent. The recent separate copy/semantic-role apprenticeship also did not generalize: its held-out result was 20/120 for the verified arm, 19/120 for its frozen parent, and 8/120 for the shuffled-answer control; verified per-skill results were copy 1/20, selection 0/20, recipient 0/20, giver 0/20, color 9/20 and quantity 10/20. That synthetic run stays a separate research branch and was not merged into this 28k language checkpoint.

The next work should keep these two learning problems separate: continue language learning with a broader, fully audited public corpus and independent generation review; build the planned question-only benchmark with independently authored wording and negative examples; and compare plain and contrastive question-intent training only after its held-out language families are sealed. Do not respond to another negative transfer result by increasing updates on the same templates.

**Claim boundary:** this is measured early language-model progress on a small public story corpus. It is not preschool-level language, robust reading comprehension, instruction following, personal reasoning, or a ready-to-integrate assistant.
