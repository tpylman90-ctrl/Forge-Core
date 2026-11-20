# Kaggle GPU candidates → GitHub draft checkpoints

Kaggle is the temporary compute surface. Qualified checkpoint bundles are
stored in a **draft GitHub Release** for Forge Core; Kaggle working storage and
Actions artifacts are not the long-term checkpoint archive. Draft status keeps
the asset out of public release publication until it is reviewed.

## Export gate

`scripts/kaggle_release_export.py export` refuses an export unless all of these
checks pass:

- The run is an isolated `language`, `blender`, or `godot` candidate with its
  parent checkpoint, optimizer-resumable checkpoint, lineage, tokenizer,
  source manifest, and pinned input-bundle hash.
- The candidate changed weights and both its fixed validation loss and
  independent held-out test loss improve by at least **0.2%** over the matching
  parent.
- The core-language retention guard passes within the existing **5%** limit;
  available task checks have no recorded regression.
- The test split is distinct from train and validation. Provenance declares
  each source, revision, license, permission to train and permission to
  distribute the resulting checkpoint. Private data is explicitly excluded.

A passing run is stored only as an unpublished draft. It is not merged with
another lane, promoted into the Android app, or treated as proof of capability.
Several seeds and human task evaluation remain necessary before adoption.

## One-time Kaggle setup

1. Create a **fine-grained GitHub token** restricted to `tpylman90-ctrl/Forge-Core`
   with repository `Contents: Read and write`. Do not put the token in a
   notebook cell, this repository, or ChatGPT.
2. In the private Kaggle notebook, add a Kaggle Secret named
   `FORGE_GITHUB_TOKEN` containing that token. Read it at runtime with
   `kaggle_secrets.UserSecretsClient`; it must not be printed or written to disk.
3. Make an `export-manifest.json` beside the result bundle. It must use schema
   1 and contain `repository`, `run_id`, `lane`, `seed`, `source_commit`,
   `input_bundle_sha256`, `private_data_in_training: false`,
   `human_task_regression: false`, the reviewed `sources` list, and relative
   paths named `metrics_path`, `lineage_path`, `checkpoint_path`,
   `parent_checkpoint_path`, `tokenizer_path`, and `source_manifest_path`.
   `sources` entries need `name`, `url`, `revision`, `license`,
   `training_allowed: true`, and `weight_distribution_allowed: true`.
4. Copy `scripts/kaggle_release_export.py` into the private notebook or its
   attached research input, set the token in the process environment from the
   Kaggle Secret, then run:

   ```bash
   python kaggle_release_export.py export --bundle-dir /kaggle/working/forge_gpu_probe
   ```

   The script builds a small checkpoint-and-provenance bundle and uploads it
   directly to a unique, unpublished GitHub draft release. It does not include
   raw training text. Re-fetch the pinned, reviewed source later and verify its
   hashes before resuming.

## How a saved candidate is used

Review the draft's `qualification.json`, `metrics.json`, `lineage.json`, and
`SHA256SUMS.json`. To resume, download `candidate-checkpoint.pt`, verify its
SHA-256 and lineage against the matching lane, seed, tokenizer, and pinned data
revision, then use it as that same lane and seed's parent. Keep the original
parent. Do not use a Blender candidate for Godot, language, visual, or another
seed. A draft is research storage; promotion or publication remains a separate
review decision.

After reviewing a specific draft, resume it in Kaggle with the helper and
the exact release tag:

```bash
python kaggle_release_export.py download \
  --tag forge-kaggle-blender-s43-run-name-20261120t120000z
```

This only retrieves unpublished drafts, verifies every bundled SHA-256 entry,
and writes the checkpoint under
`/kaggle/working/forge-github-import/candidate/forge-kaggle-candidate/`.
Use `candidate-checkpoint.pt` only as the parent for that same lane, seed,
tokenizer, and pinned dataset revision. Keep `parent-checkpoint.pt` beside it
for lineage comparisons.

The currently tested Kaggle result does **not** meet this gate: the Blender
independent test worsened and all Godot runs kept the parent weights. Those
files should remain unexported. Visual training requires its own independently
scored gate before this exporter accepts that lane.
