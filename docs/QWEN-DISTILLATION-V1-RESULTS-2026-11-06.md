# Qwen3 language distillation v1 — results (2026-11-06)

## Result

The matched study completed successfully in GitHub Actions run [42247943454](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42247943454). Each arm made 8,000 updates from the same verified 95,000-step Forge Seed parent, resuming its optimizer state and RNG. Both retained Forge's tokenizer and 2,219,136-parameter architecture.

**The 10% Qwen mixture did not improve general language prediction.** It reduced loss on held-out Qwen-generated text, but worsened both TinyStories validation and independent test loss. The uniform arm made a very small improvement on the independent test set. Neither arm showed capability improvement.

| Arm | Validation loss (parent → candidate) | Independent test loss (parent → candidate) | Held-out Qwen text loss (parent → candidate) |
|---|---:|---:|---:|
| Uniform TinyStories | 3.397505 → 3.393130 | 3.371135 → 3.367818 | 5.382249 → 5.348601 |
| 10% Qwen / 90% TinyStories | 3.397505 → 3.424190 | 3.371135 → 3.400050 | 5.382249 → 3.977293 |

Lower loss is better. Relative to its parent, the Qwen arm's independent test loss increased by **0.028915** (about 0.86%); the uniform arm's decreased by **0.003318** (about 0.10%). The Qwen arm's strong held-out synthetic-text gain shows it learned to predict examples from that small teacher-generated distribution; it does not demonstrate broader language or knowledge transfer.

## Capability check

- Qwen parent: **20/40**, with **0/20** mirrored pairs fully correct.
- Qwen candidate: **20/40**, with **0/20** mirrored pairs fully correct.
- Uniform candidate: **19/40**, with **0/20** mirrored pairs fully correct.
- Generated sentences and stories were not human-graded.

These results remain narrow diagnostics, not evidence of general comprehension or reasoning.

## Provenance and artifacts

- Forge parent: run [42143390701](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42143390701), step 95,000, SHA-256 `a7bcdb996e440f44d7496b887b54d19324927b4296fe9ed2f0f0293c88c63a0a`.
- Teacher: `Qwen/Qwen3-0.6B`, immutable revision `c1899de289a04d12100db370d81485cdf75e47ca`, Apache-2.0.
- Generated targets: 192 train and 32 held-out examples; zero exact duplicate responses; SHA-256 `cfb60bde877712ca0b8777502a4208becdbde06daef3777ef06ef08524ac312e`.
- TinyStories: pinned revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, CDLA-Sharing-1.0. Train SHA-256 `81e8ef34fc61996f8cf0e783c95daa5521074b3ec8b9a7db4912445a6bb717cd`; validation SHA-256 `ed86a376fb836f86829b930857e7ce25fd70c8824c12d48c558af854154d88ef`; independent test SHA-256 `2f816a0fde9e8b6960c6e51673ec9286b6438d9866f4678e9535cfbcbdc4609e`.
- Qwen candidate SHA-256: `7ebba3cfde9f8e0dcedb813fead4948c3fc34bd6394e3ffed23e587d723898d4`.
- Uniform candidate SHA-256: `22bdb5ee91aa0869bd7210c3457ea603b7aa7b9f801c60848e3bfa834e411325`.
- [Qwen candidate artifact](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42247943454/artifacts/13522566180), [uniform control artifact](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42247943454/artifacts/13522560483), and [teacher-data artifact](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42247943454/artifacts/13520703284) are retained for 90 days. Draft research releases: [Qwen arm](https://github.com/tpylman90-ctrl/Forge-Core/releases/tag/untagged-32526370dd72ac1e4e0d) and [uniform arm](https://github.com/tpylman90-ctrl/Forge-Core/releases/tag/untagged-61b52ed40ff2cbf67022).

## Decision

Keep the Qwen-mixed checkpoint as a research artifact; do not use it as the next core continuation parent or integrate it into Android. This 10% recipe improved fit to teacher text while harming the independent TinyStories result. Continue the core language lane from its verified parent using the uniform control. Any further Qwen distillation should remain a separate experiment with a lower mixture rate and repeatable gains across seeds before broader claims.
