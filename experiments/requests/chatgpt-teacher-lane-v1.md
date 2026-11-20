# ChatGPT teacher lane v1 pilot request

Run `.github/workflows/chatgpt-teacher-lane-v1.yml` on main.

Compare a matched control arm with ChatGPT-authored lessons for the separate language, Blender, and Godot CPU branches. Use the pinned 107,250-step Forge Seed parent, one fixed seed per lane, 500 updates per arm, validation every 100 updates, and the existing shared-language retention guard for specialists.

Keep all three lesson packs separate and hash-attributed in the reports. Save the parent, candidate, teacher pack, held-out teacher loss, lane test metrics, and core-language retention results together as Actions artifacts.

This is a one-seed pilot only. Do not merge lane weights, change canonical Forge weights, publish a release, or claim tool competence from token loss. If a lane shows a useful signal, use independent-seed replication and task-level evaluation before considering a checkpoint for durable storage beside other sessions.

Retry note: this run request was re-pushed after correcting workflow expression interpolation; use the corrected workflow version.
