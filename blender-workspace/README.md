# Forge Blender Workspace

Dedicated headless Blender execution area for Forge Core. This workspace is separate from training and does not load neural weights.

## Run a job

Use the [Forge Blender Workspace workflow](https://github.com/tpylman90-ctrl/Forge-Core/actions/workflows/blender-workspace.yml), select Run workflow, and choose a registered job. A push to main changing this workspace also runs the smoke job. No Unity license, Blender account, paid API or repository secret is required.

- jobs/: JSON asset requests, not executable scripts.
- run_job.py: bounded job launcher and artifact manifest.
- ../production/: trusted Blender construction and GLB validation code.

Initial registered job: cargo_crate. Future reviewed builders and inspectors can be added to the registry; arbitrary pasted Python, URLs and shell commands are not accepted as job inputs.

## Local equivalent

    python blender-workspace/run_job.py --job cargo_crate --out output/blender-workspace

Requires Blender on PATH and Python 3.10+. The output directory must not exist. The job uses a fresh factory scene and disables automatic script execution on startup. It runs only checked-in, trusted Python.

## Output

Each successful job publishes an Actions artifact with editable asset.blend, textured asset.glb, four preview PNGs, request.json, validation.json, Blender log, and artifact_manifest.json with hashes and version information. Logs are uploaded even on failure when present. Artifacts expire after 30 days; approved assets can be archived separately. This public repo is not a location for private references or credentials.

This initial job generates UV-space base-color textures and constant metallic/roughness properties, not full PBR baking. The validation report is a structural check, not a visual quality certification. No model is promoted or trained.

GitHub runners are ephemeral, CPU-based and subject to Actions availability and quotas. A successful workflow run is required before claiming the environment is verified.
