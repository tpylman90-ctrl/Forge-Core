# Run ChatGPT adaptive tutor v2

Run `.github/workflows/chatgpt-tutor-v2.yml` on main.

This is a follow-up to the controlled Ask-Why pilot. Train the tutor-v2 examples for language, Blender, and Godot using two reproducible seeds per lane, all restored from the same verified 107,250-step parent. Compare each result against held-out tutor-v2, direct-teaching, Ask-Why v1, lane, and shared-language evaluations.

Tutor-v2 cards must include a plausible initial mistake, a diagnostic question before the correction, explicit misconception diagnosis, short stepwise explanation, analogy, reusable principle, faded retrieval, a changed transfer problem, and an observable check. Keep lanes isolated, preserve exact parent/data/candidate/metrics, and keep candidates research-only. Do not promote weights based on next-token loss.
