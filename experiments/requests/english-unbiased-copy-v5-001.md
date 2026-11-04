# One-time Forge English v5 — aggressive unbiased copy pilot

We are continuing Forge Core's own original scratch-trained language
network education, **not** deploying third-party model intelligence.

The completed v4 letter-position study achieved 214/498 generated
letters for its mixed-taught policy and outperformed shuffled teaching,
but the exam was strongly biased: every second position was a vowel.
Exact externally assembled words succeeded only 3/96 times.

Authorize this bounded corrective research pilot:

- Reuse and cryptographically source-verify the original v0.7
  scratch-trained Forge Seed language model from run 41572257363.
- Train on 180 novel source families per seed, with independently
  random a–z letters (length 4–8), including a **single-character
  changed counterfactual twin** for every family.
- Test the neural model's generated characters at **every position**
  in two variants of 40 fresh, separately hashed final families.
- Report correct answers for indices 3–8, both-twin-correct
  hard pairs, source-sensitive changed predictions, and externally
  assembled full words; avoid mistaking assembly for model output.
- Compare four **equal-update** controls: exact verified twins
  paired per batch, same examples shuffled across batches, labels
  shuffled within position, and masked source text with truthful
  but unanswerable labels.
- Two independent training seeds, 11631 and 11632, **900 optimizer
  updates per arm per seed**, exactly 4 examples per step; 20%
  original public-story rehearsal. Run 8 models sequentially in
  one GitHub CPU workflow, timeout **45 minutes**, without retry
  loops or recurring compute.
- Choose weights on dev-only cases, and inspect new final synthetic
  exam **once**. Preserve all negative results and early curves.
- Score held-out original public-text loss on identical 48 × 64-token
  validation windows to monitor language retention.
- Keep all model states explicitly experimental and unpromoted;
  do not alter neural pretraining checkpoints, `android/`, APK
  signing, user observations or encrypted personal memories.

If the unbiased twin test fails, we should redesign the original
model's source-index representation before running longer copies
of the same training experiment.
