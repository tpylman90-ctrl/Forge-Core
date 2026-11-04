# One-time Forge Seed v3 — writing unfamiliar letter sequences

Following v2's verified negative result (all eight trained
generative English arms scored 0/112 exact held-out answers),
test **copying variables and characters** before increasing
sentence-training complexity.

- Keep original scratch-trained Forge Seed v0.7 model/tokenizer from
  source-verified research run 41572257363; no pretrained AI used.
- Use four independently verified synthetic skills: copy one
  lowercase letter, copy 3–6 spaced characters, join characters
  into an unfamiliar word, and append a new final letter.
- Whole output sequences are SHA256-partitioned between
  training, development and first-use examination sets.
  Exam uses independently generated strings and alternative wording.
- Teach via actual generated answer tokens, then evaluate using
  greedy output **exact accuracy, first-character correctness,
  character edit distance, and answer termination**.
- Same architecture, source, optimizer budget and training examples
  for ordered and mixed lessons, with shuffled-answer and
  public-story-only controls.
- Run two independent training seeds (9231 and 9232), 120 optimizer
  updates per four curriculum stages, four arms per seed,
  on a **single serialized CPU runner**, 45-minute job ceiling.
- Preserve matched 48-window held-out original-story language loss.
  Training-only improvements do not qualify as skill transfer.
- Archive experimental-only weights and metrics; **no automatic
  promotion** to the main Forge language checkpoint.
- Do not touch the concurrent Android app, signed APK workflow,
  external actions, the user's phone data or encrypted memories.
- This text exercise is not speech, pronunciation or independent
  conversational understanding.

The first result must be a controlled diagnosis of whether our
original BPE Transformer can carry unfamiliar characters into
its own generated answers.
