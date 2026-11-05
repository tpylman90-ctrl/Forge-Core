# Forge Seed — 50 MiB corpus, 28,000-step results

**Completed:** November 5, 2026  
**Training run:** [42085634121](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42085634121)  
**Checkpoint artifact:** [forge-seed-v07-research](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42085634121/artifacts/13445481872) (expires December 5, 2026)

## Run design

This is an independent random-initialized model, not a resume of the narrow 10 MiB model. It used the pinned TinyStories revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64, the declared CDLA-Sharing-1.0 license, a scratch-trained 2,048-token BPE tokenizer, seed 43, 28,000 AdamW updates, batch 4 and sequence length 128. Source streaming was bounded to 50 MiB from the training file and 5 MiB from the separate validation file; exact duplicates and train/validation overlaps were screened.

The cleaned bounded split contained **58,028 training stories** and **6,246 validation stories**. The 2.22M-parameter model's within-run held-out token loss decreased from **7.617847 to 2.363302**. This value must not be compared directly with the narrow model's token loss because corpus size and tokenizer are different.

## Held-out skill probe

| Measure | Result | Provisional criterion |
| --- | ---: | ---: |
| Mirrored A/B question accuracy | **21/40 (52.5%)** | At least 32/40 |
| Both answer-order mirrors correct | **2/20 pairs** | Stable counterfactual evidence |
| Color | 6/10 |  |
| Count | 5/10 |  |
| Location | 6/10 |  |
| Recipient | 4/10 |  |
| Single-step instructions | Not measured | 8/10 |
| Human sentence/story rating | Not formally measured | 8/10 sentences; 3/5 linked stories |

The narrow 28,000-step model scored 19/40 on the same QA diagnostic. The +2 answers here are a small difference on only 40 template-based choices, not enough to claim reliable improvement or semantic understanding. Both models are below the 32/40 gate; the recipient category remains weakest. The automatic audit explicitly does not certify prose or instruction-following.

## Generation review

The six fixed-seed continuations show recognizable English clauses and sometimes plausible local action. They still contain invented or malformed words, pronoun/character confusion, repeated entities and abrupt event changes. This small qualitative inspection is not a blind human-rated sentence/story evaluation.

## Decision

**No Friday readiness pass and no checkpoint promotion.** The wider text prefix improved breadth of exposure and next-token fitting, but the held-out QA movement is small and the generation samples remain inconsistent. Keep this checkpoint separate from the previous 10 MiB model and all task heads.

The question-only benchmark has passed its split-integrity check. A separately authorized three-seed study is now evaluating plain versus contrastive question-intent heads over this checkpoint's frozen language representations, with a balanced shuffled-label control. Even a positive result will remain a narrow diagnostic; it cannot certify story reading or instruction following.
