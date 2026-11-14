# Retry blender guided micro-runs after pre-update hash fix

The first attempt stopped before any optimizer update because the runner did not install NumPy and the stage-hash helper relied on it. The helper now hashes CPU tensor bytes without NumPy. Verify the exact parent/corpus as before, then run the same three seeds and staged schedule. Preserve separate per-seed metrics and the lane summary draft archive.
