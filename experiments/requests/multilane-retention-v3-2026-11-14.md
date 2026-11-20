# Approved isolated three-lane CPU training continuation

Train the language, Blender, and Godot lanes independently from the verified
107250-step Forge Seed parent in run 42552572467. Run at most 5,000 updates or
30 minutes per lane. Use the original pinned language split, the versioned
licensed Blender and Godot documentation corpora, and lane-local optimizers
and checkpoints. Do not merge the specialist weights into language weights.

Evaluate and save a resumable best checkpoint every 250 updates. For Blender
and Godot, keep a candidate only when its specialist validation improves and
shared-core language loss remains within +0.05 of its parent. After six
consecutive evaluations without an eligible improvement, stop that lane and
retain its best checkpoint. For language, continue from its parent optimizer
and RNG state, with no specialist replay.

Record attempted updates separately from selected checkpoint updates. Preserve
the verified parent, candidate, source manifests, data hashes, validation and
test scores, and per-checkpoint curves in each lane artifact. All three runs
are CPU-only and remain isolated research candidates. Do not promote weights,
merge lanes, or integrate Android automatically.
