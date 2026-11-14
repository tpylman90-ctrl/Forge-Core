# Guided Three-Lane Curriculum v3 — Results (2026-11-14)

## Run outcome

Nine isolated research candidates completed: three seeds each for language, Blender 5.4, and Godot 4.7. Every candidate completed all three 500-update stages (1,500 updates per seed), with fixed validation every 100 updates. The initial attempt failed before any optimizer updates because NumPy was absent from the minimal runner image; the stage-hash code was corrected to hash tensor bytes without NumPy, and the full nine-seed set was rerun.

All candidates came from the same verified Forge Seed v0.7 parent, run 42552572467, step 107250, checkpoint SHA-256:
`848553a81549c1269c360c5d07c57adde92a7816a0bc19b61ecc8028c93fee99`.

TinyStories was pinned to revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, declared license CDLA-Sharing-1.0. Tokenizer SHA-256: `4ecd7803d4c369e80bda2e74c30cfb769b7e3a942f931df83383eec9f794f04c`. Each lane's train/validation/test hashes were identical across its three seeds. Blender and Godot manual page hashes and attribution are recorded in each source manifest and draft archive.

## Results

Numbers below are averages of the three seed-specific losses. Lower is better. The candidate test split was evaluated only after checkpoint selection.

| Path | Held-out validation, parent → candidate | Held-out test, parent → candidate | Shared TinyStories check |
|---|---:|---:|---:|
| Language | 2.00255 → 2.00189 (−0.00066; 0.033%) | 1.99216 → 1.99114 (−0.00103; 0.052%) | same metric |
| Blender | 5.57864 → 4.11218 (−1.46647; 26.3%) | 5.49547 → 4.12271 (−1.37277; 25.0%) | loss +0.03891 on average |
| Godot | 5.49405 → 4.49177 (−1.00228; 18.2%) | 5.77432 → 4.45148 (−1.32284; 22.9%) | loss +0.03841 on average |

All three seeds improved validation and held-out test loss in each lane. All Blender/Godot seeds finished all stages; their shared-language regression stayed below the configured +0.05 stop/selection guard. That is a measurable but nonzero language cost. The balanced TinyStories replay substantially limited the regression seen in the previous specialist experiment, but it did not eliminate it.

These are next-token prediction results on fixed corpora. The language improvement is very small. Blender/Godot results show better prediction of the selected documentation and CC0 lesson text; they do **not** demonstrate that Forge can perform useful Blender or Godot tasks. Generated specialist samples were not human-graded. No capability certification, weight merging, or Android integration occurred.

## Candidate checkpoint hashes

| Path | Seed 43 | Seed 53 | Seed 73 |
|---|---|---|---|
| Language | `9b335d23f37aaad0360a991041d51c1aa87f75ebc70fd0948568fc55a658c009` | `d680787071eb487408a012e641503d186d75cd0d018c441238238a6875c278fe` | `82f5c8713e212e319b00aa10b9b85db8b9262e6984c1b0d31333963276596e84` |
| Blender | `728996798c5cffcd93920b89b359da40a91c3182b9ab3807f5b879624c87ea39` | `17883e5df6c46e73c0aca4d4cb664bbdcf457478fa10cdf3b6611d0d5568685d` | `b9ff0faf3b620210648c89dbeb8297d5f8d7533583718a3924c17ea16c3088c1` |
| Godot | `aef34d5ee226923d6ffc64e53f2d9ac6c939f4ffd0cd3f5592bb52b33256cb58` | `fe36dcbbca5fc147aca4a7b8cfd8fb09f017e165e2772424380576b05efaaa5f` | `2195679ae59cbd72407729ad54a16e2f584c4acf87057e53b945ab0d298d90c0` |

## Runs and durable research drafts

- [Language run 43553138349](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/43553138349) · [draft archive](https://github.com/tpylman90-ctrl/Forge-Core/releases/tag/untagged-0e3f3df7e2cc35d45af7)
- [Blender run 43553140852](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/43553140852) · [draft archive](https://github.com/tpylman90-ctrl/Forge-Core/releases/tag/untagged-8837946c720798b58709)
- [Godot run 43553141871](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/43553141871) · [draft archive](https://github.com/tpylman90-ctrl/Forge-Core/releases/tag/untagged-ffe6037ec9c2f32ba62e)

Draft archives and Actions artifacts preserve each seed checkpoint, its exact parent and source/evaluation manifests. They are research-only and are not Android upgrades.
