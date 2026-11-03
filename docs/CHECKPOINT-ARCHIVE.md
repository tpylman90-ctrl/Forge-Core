# Reproducible checkpoint archive

Long CPU experiments use GitHub Actions artifacts for **temporary** working
storage. Artifacts expire after 30 days, and users must not treat them as
permanent memory. The checkpoint archive workflow provides a **manual or
explicitly requested GitHub Release DRAFT** retaining a reviewed, public-data
research experiment beyond short-lived artifacts.

## Safety requirements

- Only an experiment with the properly attributed public TinyStories preview
  is accepted by `forge_core.neural.archive`. No user conversations, private
  memories, app-use logs or secrets are included.
- The archive auditor checks checkpoint weights and optimizer existence,
  resumed step, source/license notice, loss measurements, source/manifest
  checksums, training and validation text consistency.
- A release tag must be unique. Existing releases are never replaced.
- Initial releases are **drafts**; publication requires a separate conscious
  GitHub review. The draft is not a substitute for a separate secure backup.
- A public repository model release is **not private** once published. Never
  use this workflow for personal-memory-trained models or sensitive material.

## Manual procedure

1. Review the successful language experiment's metrics and generated text.
2. In GitHub Actions, select **Archive reviewed Forge Seed checkpoint
   (draft)** and provide the successful training run ID and a unique tag.
3. Choose `approve-draft-public-data-archive` explicitly.
4. The workflow validates the downloaded artifact and creates a draft Release
   containing weights, optimizer state, data provenance, a source notice, the
   exact bounded public corpus and audit hashes.
5. Review in GitHub Releases before publishing or copying to another store.

The initial Run 003 may be archived after review by the explicit
`experiments/requests/archive-run003.md` request file. This mechanism is
intentionally one-shot and never scheduled.
