# Forge Seed — one-time wider-corpus language experiment

Run a bounded scratch-initialized language experiment using the pinned TinyStories revision and only public source text.

- Stream the first 50 MiB of the pinned training file and first 5 MiB of its separate validation file. Keep the existing complete-story parsing, cross-split duplicate screening, SHA256 manifest, and license notice.
- Train a new tokenizer from scratch on the training split only; initialize a new 2.22M-parameter seed Transformer from random weights.
- Use seed 43 and 28,000 updates, batch 4, sequence length 128, as configured in the one-time workflow.
- Do not resume or modify the 28,000-step narrow-corpus checkpoint; this is an independent model/corpus comparison.
- Preserve the new checkpoint and source manifest as a separate research artifact.
- Report loss only within this run. Do not compare BPE token loss directly to the other tokenizer/corpus, and do not promote on loss alone.
- Run the existing capability audit; separately review fixed prompt generations and report the 40-choice QA and mirror-pair result.
- No pretrained model weights, teacher labels, private data, Android memory, or automatic promotion.
