# Forge GPU training — scalable CAD and scan lane

The baseline has 18 original CC0 Forge Basics CAD-style meshes. The GSO source is pinned to b944bf0e56d93c82652b8b7243187be237434ed4 (CC BY 4.0; per-object metadata screened; attribution retained). See https://huggingface.co/datasets/hugging-3d/google-scanned-objects and https://github.com/tpylman90-ctrl/Native-model-forge/tree/c2f050a54418d69a8ddc082d10110fb4d55e985d.

| Preset | CAD variants per object | Scan variants per object | Scan shards | Maximum sample count if all 1,030 scans accepted |
|---|---:|---:|---:|---:|
| small | 256 | 16 | 4 | 4,608 + actual scans × 16 |
| medium | 512 | 32 | 43 | 42,176 |
| large | 1,024 | 128 | 43 | 150,272 |

Variants are rescaled/rotated samples, not new base meshes. Actual counts may be smaller due to license filtering, invalid meshes or duplicate IDs/bytes. Near-duplicates across distinct scan IDs remain a limitation. No third-party data without reviewed terms. ABC CAD and Poly Haven are future intake candidates; ABC creator rights and Poly Haven bulk/API terms must be reviewed before ingestion.

## Kaggle execution

Import [forge-scaled-cad.ipynb](forge-scaled-cad.ipynb) into a private Kaggle notebook, enable a free GPU and Internet, then Run All. The notebook uses pinned trainer commit 4d3d885609361bfa23e7f82b63543167a6ff51d1; default small preset, 8 epochs and seeds 43, 53, 73, with early stopping. Increase presets only after verifying runtime, storage, accuracy and remaining quota. No paid runners. The first notebook execution has not yet been verified.

Disk-backed uint8 voxel arrays avoid keeping the entire dataset in RAM. Source objects are split before augmentation. CPU geometry preprocessing is not GPU accelerated. The target is sampled 32³ surface voxels from silhouette inputs, not a textured GLB or arbitrary-photo reconstruction. Compare against visual-hull baselines. Each candidate saves weights, optimizer state, per-epoch validation and final held-out results. Preserve all evidence; do not promote on a single seed.

Kaggle output includes the evidence ZIP. Existing legacy GitHub exporter does not support visual candidates. Durable draft-release transfer for visual results requires a separate validated exporter or reviewed manual archive. No credentials in code.
