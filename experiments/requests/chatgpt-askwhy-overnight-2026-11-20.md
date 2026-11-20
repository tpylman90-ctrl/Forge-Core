# ChatGPT Ask-Why overnight candidate run — 2026-11-20

Run four isolated tutor-v2 candidates from the pinned Forge Seed v0.7 step-107,250 parent:
- Language seeds 43 and 44 (independent seed replication)
- Blender seed 731
- Godot seed 947

Each candidate receives up to 120 minutes and 50,000 updates, evaluates every 100 updates, and stops after three evaluations without lane-validation improvement. The trainer's core-replay and 0.05 absolute regression gate remain enabled. Candidates are retained per lane and seed with parent, tokenizer, source and tutor lesson files, held-out evaluation, and metrics.

Purpose: extend the Ask-Why training format where Forge attempts a task, asks why, receives a causal breakdown, retrieves the principle, transfers it to a changed case, and checks the result. This is a controlled training candidate study, not evidence of improved general intelligence or practical task capability. No candidate is promoted, merged across lanes, or installed automatically. The active Kaggle visual pilot remains a separate run.
