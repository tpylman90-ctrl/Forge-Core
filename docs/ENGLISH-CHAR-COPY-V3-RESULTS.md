# Forge English v3 — verified character-copy results (2026-11-04)

**Research-only. No checkpoint promotion.**

[Successful two-seed CPU experiment](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41837615641) at commit `bb5ac8c65082d26a469d5e64821015051cc66799`. Original audited scratch TinyStories source: [run 41572257363](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41572257363). Research artifact **13327112484**; no pretrained weights, personal observations, or Android modifications.

The study used two independent training/schedule seeds, 9231 and 9232, with 480 bounded updates per arm and seed. The four arms were verified staged, verified mixed (same examples in shuffled time order), shuffled-answer staged, and TinyStories-only. Every arm began from the same original 14,000-step scratch-trained Forge Seed v0.7. The synthetic exam contained **96** never-trained examples per seed (24 each: copy one letter, copy spaced sequence, join letters, append letter), with distinct multi-character strings assigned by SHA256 bucket. Exam prompts used different wording but shared the same task structures. This exam has now been inspected and is no longer untouched.

| Seed | Verified staged | Verified mixed | Shuffled staged | Stories only |
| --- | ---: | ---: | ---: | ---: |
| 9231 exact /96 | **15** | 9 | 0 | 0 |
| 9232 exact /96 | 4 | **12** | 0 | 0 |
| Both exact /192 | **19** | **21** | 0 | 0 |

The unadapted original scored **0/96** on the exam. First-character correctness (out of 96 per seed) was 86/91 for staged/mixed seed 9231 and 66/82 for staged/mixed seed 9232. Shuffled controls scored 5 and 4 first characters, while story-only controls scored 3 and 0. This shows **some character-level learning**, not reliable copying of complete unfamiliar strings. Staged lessons did not consistently outperform mixed lessons. Error counts for staged/mixed were 229/224 for seed 9231 and 250/260 for seed 9232; these are Levenshtein edit counts, not percentages.

Held-out TinyStories story-loss ratios relative to the frozen source: verified staged **1.04611 / 1.03199** and verified mixed **1.02718 / 1.03288** (seeds 9231/9232). All are within the preliminary 5% loss-regression gate, but worse than the original. Shuffled controls had ratios 1.00964 / 1.04048 and story-only controls 0.99465 / 0.99974.

**Decision:** retain the original archive and research artifacts without promotion. V3 demonstrates first-character progress over controls, but 19/192 and 21/192 exact copied strings are insufficient for reliable text copying, preschool readiness, or grammatical transfer. The next separate experiment should target *multi-character copying* using explicit source-aligned character decoding and a genuinely fresh independent exam; do not blindly extend this exact training schedule or retune against the already-used v3 exam. Include matched incorrect-label and language-retention controls.
