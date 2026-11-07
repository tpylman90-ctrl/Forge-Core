# Forge small specialist learning sweep

Run the eight short CPU trials described in .github/workflows/specialist-small-sweep.yml:

- Lanes: Blender and Godot; each starts from the same verified Forge Seed v0.7 95k checkpoint (run 42143390701; checkpoint SHA-256 a7bcdb996e440f44d7496b887b54d19324927b4296fe9ed2f0f0293c88c63a0a).
- Compare the prior specialist learning rate (5e-5) with a lower rate (1e-5).
- Run two independent sampling seeds per lane and rate.
- Cap each arm at 3,000 updates or 15 minutes, whichever comes first; evaluate held-out documentation every 250 updates and stop after three evaluations without improvement.
- Preserve each lane, seed, and rate in its own checkpoint and artifact with hashes, attribution, losses, and stop reason.
- Keep all weights research-only. Do not merge specialist branches into each other or into language weights; do not integrate into Android.
- Generated task samples are not graded, so report no tool-competence certification.

This is a short hyperparameter and seed sensitivity experiment motivated by the previous long run selecting its best validation checkpoint at 1,000 updates and worsening afterward.