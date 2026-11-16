# Retry language guided micro-runs after pre-update hash fix

The first attempt stopped before any optimizer update because the runner did not install NumPy and the stage-hash helper relied on it. The helper now hashes CPU tensor bytes without NumPy. Verify the exact parent/corpus as before, then run the same three seeds and staged schedule. Preserve separate per-seed metrics and the lane summary draft archive.


## Follow-up run — 2026-11-16

Request a fresh bounded follow-up run on 2026-11-16. Keep the verified 107250 parent, fixed data split, three seeds, staged training, and existing validation and retention guards. Do not promote weights.
