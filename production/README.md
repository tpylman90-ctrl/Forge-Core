# Forge production prototype v1

The first bounded asset family is a modular science-fiction cargo crate. This is deterministic procedural construction, not a neural generator or a claim of professional quality.

Run with Blender installed (tested CI target Ubuntu 24.04 package):

    python -m unittest discover -s production -p 'test_*.py' -v
    blender --background --factory-startup --python-exit-code 1 --python production/build_blender.py -- --request production/requests/cargo_crate.json --out output/crate
    python production/validate_glb.py output/crate

Use a fresh output directory. Request schema version 1 accepts dimensions in meters, RGB color, seed, triangle budget and base-color texture size. It accepts no arbitrary scripts, URLs or file paths. Only cargo_crate is currently supported.

Outputs: editable asset.blend with packed generated base-color textures, asset.glb, four fixed preview PNGs, request.json and validation.json. Metallic and roughness are PBR constants; generated UV-space base-color maps are embedded. This is not a full texture-baking pipeline; normal, AO, roughness and metallic texture baking are future work.

The independent parser checks binary structure, embedded image signatures, textured primitive references, UVs and triangle count. It does not certify geometric validity, dimensions, appearance, full glTF specification conformance, or engine compatibility. Visual approval and further inspection remain separate gates.

CI should produce a downloadable bundle, not automatically publish assets or weights. No training or proprietary model is required. Do not reuse untrusted Blender scripts or files in the worker.
