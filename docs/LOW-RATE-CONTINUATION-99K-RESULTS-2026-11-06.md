# Forge Seed v0.7 low-rate continuation — three-seed results (2026-11-06)

## Result

Three matched sample-order seeds completed 4,000 uniform TinyStories updates from Forge's verified 95,000-step scratch-origin checkpoint. The saved AdamW moments were resumed and the learning rate was reduced from 0.0003 to 0.0001. Validation was checked every 1,000 updates; all three seeds improved at every check, so each completed all 4,000 updates and selected its 99,000-step candidate.

**The lower-rate continuation produced a repeatable next-token prediction improvement on this fixed data evaluation across all three sample orders.** The untouched TinyStories range loss decreased in every seed. This is evidence of improved prediction on these fixed text samples—not evidence of comprehension, reasoning, or instruction following.

| Sampling seed | Validation loss (95k → 99k) | Untouched test loss (95k → 99k) | Candidate checkpoint SHA-256 |
|---:|---:|---:|---|
| 43 | 2.11754341 → 2.03893377 (−0.07860964) | 2.11030750 → 2.02956198 (−0.08074552) | `4ab13c95116680c1c8ee9f94c83c4b9550f65129a6bef146da41d0e4d68bd7ca` |
| 73 | 2.11754341 → 2.03671362 (−0.08082979) | 2.11030751 → 2.02716126 (−0.08314625) | `b1ba557fc08b2a93c49857499f8d5a13c76eb7a5eb78a94254018876350631ad` |
| 101 | 2.11754341 → 2.03750858 (−0.08003482) | 2.11030750 → 2.02917139 (−0.08113610) | `fe3c6d5fc73500cfb372bc2f4e0fbb35e45f7a99a5420c91795a3aaf16953053` |

The mean independent test change was **−0.081676**, about **3.87% lower loss** from the parent. The evaluation uses 2,048 evenly spaced 128-token windows on the same pinned validation split and the same untouched TinyStories byte-range test. These 128-token results must not be compared numerically with the earlier Qwen experiment's 256-token-window evaluation.

## Capability diagnostics

The existing 40-choice forced-A/B diagnostic remained at chance:

| Seed | Parent | 99k candidate |
|---:|---:|---:|
| 43 | 20/40; 0/20 mirrored pairs fully correct | 20/40; 0/20 fully correct |
| 73 | 20/40; 0/20 fully correct | 20/40; 0/20 fully correct |
| 101 | 20/40; 0/20 fully correct | 20/40; 1/20 fully correct |

On the separate fresh mirrored probe, the candidate scores were 20/40 (seed 43), 20/40 (seed 73), and 19/40 (seed 101); none had a fully correct mirrored pair. Both checks are small first-token A/B diagnostics and do not establish comprehension or reasoning. Generated sentences and stories were not human-graded.

## Provenance

- Parent: run [42143390701](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42143390701), step 95,000, SHA-256 `a7bcdb996e440f44d7496b887b54d19324927b4296fe9ed2f0f0293c88c63a0a`.
- Seed 43 run: [42287595328](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42287595328); artifact [13537614483](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42287595328/artifacts/13537614483); archive [draft release](https://github.com/tpylman90-ctrl/Forge-Core/releases/tag/untagged-81191b077e26e647e8c8).
- Seeds 73 and 101: [matched replication run 42289055100](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42289055100); artifacts [seed 73](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42289055100/artifacts/13538174812) and [seed 101](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42289055100/artifacts/13538279292). Each seed was also preserved as a draft research release by the workflow.
- Model: original 2,219,136-parameter Forge Seed v0.7 and original tokenizer; no teacher weights or personal data.
- Data: TinyStories revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, CDLA-Sharing-1.0; train and validation split hashes match the verified parent. The independent 4 MiB byte-range test was held out from training and validation.
- Training: CPU, batch 4, context 128, 4,000 updates, LR 0.0001; optimizer moments resumed; sampling-order seeds 43, 73, and 101.
- CI passed on the replication change. One preliminary seed-43 attempt completed training and evaluation but stopped during capability packaging because the parent evaluation directory lacked the source manifest. The corrected deterministic rerun succeeded and archived the checkpoint and diagnostics.

## Decision

The low-rate uniform recipe is the current leading language-training method: it showed the same direction of held-out text-loss improvement across three sample-order seeds. Keep these 99k candidates as research checkpoints; do not promote them to Android or call them a capability upgrade. The chance-level probes show that language prediction is improving while simple forced-choice performance has not.

Next, continue cautiously from the validation-best 99k candidate at a lower rate and a bounded update budget, while evaluating on a fresh TinyStories range. Keep capability evaluation separate and require human review of generated text before making any capability claim.
