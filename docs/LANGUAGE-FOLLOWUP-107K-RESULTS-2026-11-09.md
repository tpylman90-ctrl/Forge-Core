# Language continuation from the selected 107k checkpoint — 2026-11-09

## Run and lineage

Workflow [42552572467](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/42552572467) continued Forge Seed v0.7 from the validation-best 107,000-step checkpoint selected in sampler-43 arm of run 42539881351.

- Parent checkpoint SHA-256: `ed2f1395b52c3ae2c674e00f3b86f82ec26685e2d1c631f6eb975ddb3f05fe0f`
- Parent validation loss (2,048-window evaluation): 2.00333783
- Parent independent-test loss (2,048-window evaluation): 1.96508341
- TinyStories revision: `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, license `cdla-sharing-1.0`
- Train SHA-256: `81e8ef34fc61996f8cf0e783c95daa5521074b3ec8b9a7db4912445a6bb717cd`
- Validation SHA-256: `ed86a376fb836f86829b930857e7ce25fd70c8824c12d48c558af854154d88ef`
- Independent-test SHA-256: `9d3ad642916b7b4494efae808ce6683b21e8b8eed3778ade934cf2924b86ea0f`
- Tokenizer SHA-256: `4ecd7803d4c369e80bda2e74c30cfb769b7e3a942f931df83383eec9f794f04c`

The run resumed the parent optimizer moments and saved sample RNG, kept the same data, and used the same conservative learning rate (1.25e-5). Maximum requested updates were 1,000; early stopping occurred after 750 when validation had failed to beat its best result twice.

## Results

| Checkpoint | Global step | Validation loss | Independent-test loss | SHA-256 |
|---|---:|---:|---:|---|
| Parent | 107,000 | 2.00333783 | 1.96508341 | `ed2f1395b52c3ae2c674e00f3b86f82ec26685e2d1c631f6eb975ddb3f05fe0f` |
| Selected candidate | 107,250 | 2.00255106 | 1.96464099 | `848553a81549c1269c360c5d07c57adde92a7816a0bc19b61ecc8028c93fee99` |
| Last checkpoint (not selected) | 107,750 | 2.00263317 | 1.96402446 | `a7f8747b70fc47aed85847f114e104295318d4c97a3de7aacdda0420e8d3d255` |

The selected candidate reduced validation loss by 0.00078677 and independent-test loss by 0.00044242 versus its parent. The last checkpoint had lower independent-test loss but worse validation loss than the selected candidate, so validation-based selection retained step 107,250. This is a small fixed-sample next-token prediction gain, not evidence of comprehension or reasoning.

## Capability checks

Parent and selected candidate both scored 20/40 on the forced-choice probe, with 0 of 20 mirrored answer pairs both correct. Fresh A/B checks also remained unchanged at 20 correct. Forge remains uncertified; human grading of generated stories or answers was not performed.

## Preservation and status

The selected checkpoint, parent, last checkpoint, optimizer state, tokenizer, data, hashes, and capability evaluations are preserved in the [draft research release](https://github.com/tpylman90-ctrl/Forge-Core/releases/tag/untagged-0f4dcb4f1d4a5d). The candidate is research-only; it was not promoted to Android or merged into Blender/Godot specialist weights.
