# Approved Blender lane retry

Run only the Blender specialist lane; do not restart language or Godot. The
first Blender job in run 42149964829 failed before training because one
documentation URL yielded too little text. That attempt produced no checkpoint
or training updates.

Use the corrected Blender 5.4 documentation source list, with a specific Mesh Structure page for the final test split. Verify all source hashes,
keep code blocks from the Blender Manual excluded, retain Blender Documentation
Team attribution and CC BY-SA 4.0 notice, and include the original CC0-1.0
lesson cards. Resume the Blender-only branch from the verified 95k Forge parent
(run 42143390701; checkpoint SHA-256
a7bcdb996e440f44d7496b887b54d19324927b4296fe9ed2f0f0293c88c63a0a).

Give the Blender-only lane up to 3,600 CPU seconds on a standard free public
GitHub-hosted runner. Preserve its own model, fresh optimizer, RNG, page-level
train/validation/test split, task samples, and lineage. Keep it research-only;
do not alter language/Godot weights or Android weights.
