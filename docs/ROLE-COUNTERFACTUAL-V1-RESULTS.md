# Forge Seed v0.7 — paired role-reversal results

**Completed:** November 5, 2026  
**Training run:** [42097136952](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42097136952)  
**Artifact:** [forge-role-counterfactual-v1](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42097136952/artifacts/13449280160)  
**Integrity CI:** [42097137119](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42097137119) passed.

## Question

Does a focused, direct-answer lesson help Forge bind giver and recipient roles when a transfer changes direction, and then generate the correct person's name on new question wording?

This run used the successful 28,000-step, 50 MiB public TinyStories scratch checkpoint from run 42085634121. The 2.22M-parameter language model and its tokenizer were verified and preserved. Both experimental arms started from identical weights.

## Lesson and controls

The curriculum contained 3,200 answer rows organized as 800 counterfactual episodes. Each episode had four prompts:

1. Ask who gave and who received an item.
2. Reverse the giver and recipient in the same event.
3. Ask both questions again.

Stages used 160 / 800 / 3,200 answer rows and 250 / 650 / 1,600 optimizer updates. The verified-answer arm was compared with a within-role shuffled-answer control under the same batches and public-language rehearsal.

The untouched final set contained 50 episodes and 200 direct-answer prompts. It used different names and objects, as well as story and question forms absent from training. One half used familiar give/get verbs in new constructions; the other half used the unseen verbs “transferred” and “delivered.” The final set was scored after training ended.

## Results

| Measure | Frozen source | Verified lesson | Shuffled control |
| --- | ---: | ---: | ---: |
| Exact generated names | 0/200 (0%) | **48/200 (24%)** | 0/200 (0%) |
| Giver answers | — | 20/100 (20%) | — |
| Recipient answers | — | 28/100 (28%) | — |
| New constructions, familiar verbs | — | 32/100 (32%) | — |
| Unseen verbs | — | 16/100 (16%) | — |
| Episodes with all four names correct | — | 0/50 | — |

The uniform-name reference is 1/12 (8.3%) per answer, but it is only a rough reference: the model sometimes generated malformed or unrelated names, so this is not a clean statistical comparison. This was one seed and one synthetic benchmark.

The score is above the frozen source and shuffled control, which is a useful training signal. Transfer remains weak: no episode had all four answers right, neither role reached 30% accuracy, and the unseen-verb slice was weaker. At the role-pair level, the model got both reversed giver answers correct in 0/50 episodes and both reversed recipient answers correct in 1/50.

Examples show why raw accuracy overstates the result. For “Oscar gave Rose the blanket,” asked who handed it over, Forge answered “Rose.” It answered “Rose” again for a different question on the reversed event, where “Rose” was correct. Some names were also malformed, such as “Juby” for Jude and “Mama” for Mila. The fixed story continuations stayed readable in places but still contained word errors, repetition, and story drift.

Held-out TinyStories loss improved from 2.369578 to 2.339282 (about 1.3% lower). This indicates no language-loss regression in this audit, but it does not establish broad language understanding.

## Decision and next step

**No checkpoint promotion.** The result supports a narrow claim: this one direct-answer curriculum raised exact performance on its untouched synthetic role challenge above both controls. It does not establish robust giver/recipient understanding.

The next experiment should add an explicit intermediate target such as “Giver: X; Receiver: Y” before the short answer, then test on a newly authored final set with new people, objects, and sentence forms. Include a name-selection diagnostic that isolates role binding from spelling unseen names. Replicate across seeds before treating the gain as stable. Keep the frozen source and current candidate weights separate.
