# Approved uniform continuation from the 79,000-step matched study candidate

Continue the original scratch-trained Forge Seed v0.7 language model from the
best uniform candidate in run 42133325924: sampler seed 73, step 79,000,
checkpoint SHA256
45c16830595356ac023533e4a6947115ddd4f123b1fcb3b74764a0987fe31f42.

The workflow restores the exact approved TinyStories revision, train/validation
files, tokenizer, model weights, and optimizer state; verifies the checkpoint
and source hashes; performs 8,000 ordinary shuffled next-token updates; and
compares parent/candidate using the same fixed 256-window evaluation. It also
runs the common capability check and archives the candidate as a research-only
draft release. No Android weights are changed.
