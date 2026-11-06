# Approved bounded uniform language recovery

Continue only the core language lane from the verified 95,000-step Forge Seed
checkpoint in run 42143390701 (SHA-256
a7bcdb996e440f44d7496b887b54d19324927b4296fe9ed2f0f0293c88c63a0a).

Run exactly 8,000 uniform next-token updates, resuming that checkpoint's own
optimizer and RNG. Use the corrected independent TinyStories test range after
the original 50 MiB training prefix. Compare the resulting 103k checkpoint with
the 95k parent on identical 2,048-window, 256-token independent text evaluation
and 20 fresh mirrored fact questions (40 answer orientations).

Preserve checkpoint, optimizer/RNG state, source and task hashes, metrics,
lineage, and ungraded generation samples in a draft research archive. Do not
resume from the 152k candidate, train Blender/Godot in this run, merge weights,
or promote anything to Android.
