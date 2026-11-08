# Controlled Forge night training batch

Run four tightly matched small trials for each independent lane: language, Blender,
and Godot. Keep each run's data, optimizer state, and candidate checkpoint isolated.

- Language: four 1,000-update continuations from the verified 105k Forge checkpoint,
  SHA-256 379cd45eb95e74c72a91aaa5dd68f5efb266c4626198279d2fb021a019102bf4.
  Resume Forge's optimizer/RNG, use LR 0.0000125, evaluate every 250 updates,
  keep patience 2, and evaluate on the fixed validation set plus a fresh 16 MiB
  TinyStories range.
- Blender and Godot: four 1,500-update branches per lane from the verified
  scratch-origin Forge 95k parent, SHA-256
  a7bcdb996e440f44d7496b887b54d19324927b4296fe9ed2f0f0293c88c63a0a.
  Keep the same lane-specific pinned documentation source and CC0 task/answer/why
  lesson cards across all four seeds; use LR 0.00005, eval every 250 updates,
  patience 3, and cap each job at 15 minutes.

Use seed variation only; do not change datasets or hyperparameters within a lane.
Preserve each checkpoint, optimizer, data/source hashes, training curve, and
evaluations. These are research-only branches; no candidate is promoted or
integrated into Android. Teacher method: the tool lanes include original
source-attributed lesson cards; the language runs use uniform TinyStories
continuation. No live ChatGPT weights/logits are available to the runner, and
the prior Qwen mix is not reused because its previous language trials regressed
on the independent test set.
