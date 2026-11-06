# Qwen low-dose replication — results (2026-11-06)

## Result

The six-arm matched study completed successfully in [GitHub Actions run 42265146379](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42265146379). Each candidate received 8,000 updates from the same verified 95,000-step Forge Seed parent, with its optimizer state restored. Uniform and Qwen-mixture arms were paired at sample-order seeds 43, 73, and 101.

**The 1% Qwen mixture did not produce a repeatable core-language improvement.** It reduced loss on held-out Qwen-generated targets in all three seeds, but every candidate—uniform controls included—worsened TinyStories validation and independent test loss relative to the parent. The Qwen arm had a lower mean test-loss regression than the uniform control, but this relative advantage appeared in only two of three paired seeds and does not change the fact that both methods regressed from the parent.

| Seed | Arm | Validation loss (parent → candidate) | Independent test loss (parent → candidate) | Held-out Qwen loss (parent → candidate) | Candidate checkpoint SHA-256 |
|---:|---|---:|---:|---:|---|
| 43 | Uniform TinyStories | 3.397505 → 3.404776 | 3.371135 → 3.380491 (+0.009356) | 5.382249 → 5.451530 | `321a866083761c0070ec96351c0592e6e18dfea1bdbad94b7414e7eb502769a3` |
| 43 | 1% Qwen / 99% TinyStories | 3.397505 → 3.398733 | 3.371135 → 3.373180 (+0.002045) | 5.382249 → 3.941110 | `e9a0d643084539ed8561231d063e8846cb6ee76f9cddf76e2f1d8c0472e1ec13` |
| 73 | Uniform TinyStories | 3.397505 → 3.407931 | 3.371135 → 3.382908 (+0.011773) | 5.382249 → 5.404610 | `75e04d6206ae2812d0837bf7b97ffed7d4f5ffaef32cf042594c35a611d25080` |
| 73 | 1% Qwen / 99% TinyStories | 3.397505 → 3.417153 | 3.371135 → 3.390085 (+0.018950) | 5.382249 → 3.971968 | `195c3ea225bdc9f1d24d6dec78a96f5878aaab0b5314efa28e56d0dcd26e0ac8` |
| 101 | Uniform TinyStories | 3.397505 → 3.411679 | 3.371135 → 3.389662 (+0.018527) | 5.382249 → 5.410361 | `27f9e6a30cc3bb6dd188a204f34a823add92df65466a6d29f84fac9c9697a48e` |
| 101 | 1% Qwen / 99% TinyStories | 3.397505 → 3.401862 | 3.371135 → 3.379776 (+0.008641) | 5.382249 → 3.946734 | `62cfa96dd32ad9be824d1c5fe245b304a1ed3b74401f53dbf398fde2b0d5ffd5` |

Lower loss is better. The independent evaluation used 2,048 fixed 256-token windows with the same tokenizer and splits. Across the three seeds, mean test-loss regression was **+0.013219** for uniform training and **+0.009879** for the Qwen mixture. The Qwen arm had a lower regression than its paired uniform arm for seeds 43 and 101, but a larger regression for seed 73. All six candidates were worse than the parent on the independent test set.

The Qwen arm’s consistent improvement on held-out teacher-generated text shows increased fit to that small target distribution. It does not establish transfer of general language ability. This experiment does not establish that uniform training is ineffective overall; it shows that this particular 8,000-update, three-seed comparison did not improve general held-out loss.

## Capability diagnostics

On the existing forced-choice probe:

| Seed | Uniform candidate | 1% Qwen candidate |
|---:|---:|---:|
| 43 | 21/40; 1/20 mirrored pairs fully correct | 21/40; 1/20 fully correct |
| 73 | 19/40; 0/20 fully correct | 20/40; 0/20 fully correct |
| 101 | 19/40; 0/20 fully correct | 20/40; 0/20 fully correct |

On the separate fresh probe (20 generated toy scenarios, each with mirrored A/B answer positions), all six candidates scored **20/40**, with **0/20** mirrored pairs fully correct. Both probes use first-token A/B scoring; these small diagnostics are vulnerable to position and prompt effects and do not demonstrate comprehension, reasoning, or a capability upgrade. No human grading of generated text was performed.

## Provenance and artifacts

- Forge parent: [run 42143390701](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42143390701), step 95,000, checkpoint SHA-256 `a7bcdb996e440f44d7496b887b54d19324927b4296fe9ed2f0f0293c88c63a0a`.
- Forge architecture: 2,219,136 parameters; Forge tokenizer retained. Tokenizer SHA-256 `4ecd7803d4c369e80bda2e74c30cfb769b7e3a942f931df83383eec9f794f04c`.
- Teacher: `Qwen/Qwen3-0.6B`, immutable revision `c1899de289a04d12100db370d81485cdf75e47ca`, Apache-2.0. The teacher generated targets only; no teacher weights were copied into Forge.
- Generated targets: 192 training and 32 held-out examples, zero exact duplicate responses, SHA-256 `cfb60bde877712ca0b8777502a4208becdbde06daef3777ef06ef08524ac312e`.
- TinyStories: revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, CDLA-Sharing-1.0. Train SHA-256 `81e8ef34fc61996f8cf0e783c95daa5521074b3ec8b9a7db4912445a6bb717cd`; validation `ed86a376fb836f86829b930857e7ce25fd70c8824c12d48c558af854154d88ef`; independent test `2f816a0fde9e8b6960c6e51673ec9286b6438d9866f4678e9535cfbcbdc4609e`.
- Fresh-probe task SHA-256: `bd5379774c7fc6f8fedbad5dfc8ec34a4d2e6fa611da3c3510851601f3e9928e`.
- All six checkpoint, parent, optimizer, tokenizer, data, and evaluation bundles were uploaded as run artifacts and preserved as draft research releases. Run artifacts expire after 90 days. No candidate was promoted to Android.

## Decision

Keep all six candidates as research artifacts. Do not use the 1% Qwen recipe as the next core continuation parent and do not integrate these checkpoints into Android. Return the core language lane to a verified Forge checkpoint and keep the next improvement attempt as a matched, reproducible experiment with the same fixed evaluations. Any further teacher-data experiment should remain separate until it shows repeatable gains over the uniform control without harming independent language loss.
