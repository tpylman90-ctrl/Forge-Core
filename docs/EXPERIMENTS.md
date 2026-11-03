# Forge Seed — Verified experiment ledger

## Run 001: from-scratch CPU pipeline validation

- **Date:** 2026-11-02 (US Eastern; UTC run date 2026-11-03)
- **GitHub workflow:** https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41549949410
- **Source:** original Forge Core generated demonstration sentences (NOT TinyStories)
- **Tokenizer:** byte-level (256 IDs); random-initialized Transformer
- **Model:** `tiny`, 24,928 trainable parameters (CI control, NOT Forge Seed's 10–30M target)
- **Training:** CPU, seed 42, 3 optimizer steps, batch 2, sequence length 32, AdamW learning rate 0.0003
- **Training:** 26,769 UTF-8 bytes; validation: 6,819 bytes
- **Held-out byte cross-entropy before:** 5.544311
- **Held-out byte cross-entropy after:** 5.493195
- **Relative difference:** approximately -0.92% (small improvement on fixed evaluation windows)
- **Held-out bits per byte after:** 7.925006
- **Artifact:** `forge-scratch-language-checkpoint`, GitHub artifact ID `13191740077`; retains checkpoint, AdamW optimizer state, RNG states, metrics history and dataset manifest; expires under repository retention policy.
- **Unit tests:** CI run https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41549948981 succeeded, including deterministic split/resume/mismatch tests.

### Interpretation

The training and evaluation code works under GitHub Actions, and the checkpoint artifacts were published. The model has not learned general language from this repetitive tiny fixture. This result is a reproducibility and bookkeeping milestone only.

### Next experiment

Run the bounded, license-reviewed `tinystories` source through **Actions → Forge Scratch Language Experiment → Run workflow**, then review held-out loss and save the checkpoint before its artifact expires. Scaling to 10–30M parameters and substantial corpus sizes requires separate compute planning and proper GPU access. No private personal memory belongs in these artifacts.


## Run 002: first real TinyStories preview experiment — Forge Seed v0.4

- **Date:** 2026-11-02 US Eastern (2026-11-03 UTC)
- **Workflow:** https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41554336734
- **Code SHA:** ec1307bf016551b62c8f80202b8efe8d91c1cefa
- **Source:** 200 public TinyStories training rows, 50 separate validation rows;
  dataset viewer preview, **not revision pinned** or a complete training corpus
- **Manifest:** 200 accepted training documents, 50 accepted validation
  documents, source/license, per-document IDs/hashes and split checks
- **Neural model:** original randomly initialized seed Transformer,
  **1,875,072** parameters, byte tokenizer, context 256
- **Optimization:** 20 AdamW steps, CPU, random seed 42, batch 2,
  training window 64 bytes, learning rate 0.0003
- **Data:** train 159,551 bytes; held-out validation 33,832 bytes
- **Held-out next-byte cross-entropy BEFORE:** **5.532837**
- **Held-out next-byte cross-entropy AFTER:** **3.581682**
- **Held-out bits per byte after:** **5.167275**
- **Comparison:** about **35.3%** relative held-out loss reduction over 16
  fixed, distributed evaluation windows in this one run
- **Generated sample:** contains many invalid/unreadable bytes; coherent
  language is **not** demonstrated
- **Checkpoint:** GitHub artifact forge-scratch-language-checkpoint,
  ID **13193865863**, includes optimizer, RNG state, manifest and metrics;
  expires **2026-12-03 UTC** if not copied to durable storage
- **CI:** https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41554336753
  passed the independent unit tests and smoke run

### Interpretation and limits

Forge can learn statistics of real held-out text using its own randomly
initialized weights. The small sample and 20-step run cannot establish
language proficiency, comprehension, conversation, or independent judgment.
The observed loss reduction may primarily reflect learning byte frequencies
and local patterns. The next experiment should use a longer controlled run,
a fixed/provenance-pinned corpus, more comprehensive held-out scoring, and
readable generation tests before promoting any capability claim.


## Run 003: same 1.875M Forge Seed learns more on locked public text

- **Date:** 2026-11-02 (US Eastern; 2026-11-03 UTC)
- **Run and logs:** https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41556317228
- **Commit:** ca2f7245379952d5a8acc48b41f4ca49b204a09a
- **Starting checkpoint:** Run 002, step 20. Same weights, optimizer,
  minibatch random state and model architecture; no pretrained model.
- **Data:** same 200 TinyStories training rows and 50 independent validation
  rows as Run 002. Both split SHA256 and complete source manifest SHA256
  validated against the checked-in exact-content lock. This is not an
  immutable upstream revision or complete TinyStories distribution.
- **Additional training:** 1,000 AdamW CPU steps, batch 2, context 64;
  **total steps = 1,020**, trained target bytes this run = 128,000.
- **Model:** original 1,875,072-parameter byte-level Forge Seed.
- **Fixed 16-window held-out loss:** **3.581682 → 2.214508**,
  approximately 38.17% lower over the continuation.
- **Expanded 128-window held-out audit:** **3.582403 → 2.127761**.
- **Final held-out bits per byte:** **3.194860** (fixed 16 windows).
- **Measured training wall-time:** 20.14 seconds (CPU training stage only).
- **Validation trajectory (global optimizer step, loss):**
  20: 3.581682; 120: 2.660929; 220: 2.536801; 320: 2.425791;
  420: 2.416255; 520: 2.387213; 620: 2.363225; 720: 2.301203;
  820: 2.276215; 920: 2.263386; 1020: 2.214508.
- **Generated sample:** starts "Once upon a time whas pime. Le thee tracresom..."
  with letter-like fragments, but **not coherent language**.
- **Artifact:** \`forge-scratch-language-checkpoint\`, ID \`13193808375\`,
  about 20.5 MB; expires **2026-12-03 UTC**. This artifact includes
  weights, optimizer, RNG, metrics, provenance, source notice and the exact
  public training/validation slices for resumption.
- **CI:** https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41556317233
  passed all Python unit tests and the training smoke test.

### Interpretation

This is a verified second checkpoint on the same public text. Validation
improved consistently and the model is beginning to learn byte-level English
patterns. Its sample is not fluent, and the corpus is very small. It would
be inaccurate to describe Forge Seed as understanding language, following
instructions or reasoning. The next research gate is more *diverse*,
immutable, properly licensed text and independent sample-quality benchmarks,
not adding Android features.


## Run 004: expanded TinyStories corpus, new scratch-initialized Forge Seed

- **Date:** 2026-11-02 US Eastern / 2026-11-03 UTC
- **Workflow and logs:** https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41557239198
- **Commit:** 60c2a98be062b44228d3ab0954f64b2c3306cfde
- **Starting weights:** RANDOMLY INITIALIZED (not Run 003, not pretrained).
- **Model:** original 1,875,072-parameter byte-token Transformer.
- **Public data:** 800 TinyStories train documents (709,378 UTF-8 bytes)
  and 200 separate validation documents (154,880 UTF-8 bytes).
  The source remains a live dataset-viewer preview, not immutable upstream.
  The exact bytes/manifest from Run 004 are now checked into the next run's
  **content lock**, not the public repository training text.
- **Training:** 4,000 AdamW steps on CPU, batch size 4, sequence length 128,
  learning rate 0.0003, seed 42; 2,048,000 target bytes processed.
- **Fixed 32-window held-out cross-entropy:** **5.546279 → 1.410106**.
- **Expanded 256-window held-out audit:** **5.539601 → 1.281533**.
- **Fixed-window bits per byte:** 2.034353.
- **Measured training stage wall time:** 216 seconds.
- **Loss by step:** 0=5.546279; 400=2.294955; 800=2.108493;
  1200=1.910141; 1600=1.780169; 2000=1.674432; 2400=1.588722;
  2800=1.545458; 3200=1.492737; 3600=1.461816; 4000=1.410106.
- **Fixed-seed generated text:** starts "Once upon a time who pull to
  safs. He worls man she who world..." with common words and fragmentary
  structure, but still **not coherent English**. A 1.0 printable fraction is
  NOT fluency, reasoning or comprehension.
- **Saved temporary artifact:** ID \`13194264682\`, with model weights,
  optimizer/RNG, full bounded public split, manifest and metrics;
  expires **2026-12-03 UTC**.
- **CI:** https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41557239190
  passed.
- **Next experiment:** Run 005 resumes this exact checkpoint with unchanged
  data/settings for 4,000 additional steps. Its expanded-corpus SHA lock
  fails closed on an upstream content change.

### Assessment

Forge Seed is showing substantial measured progress on byte-level language
statistics, on a substantially larger set of texts. It is still a small
research language model, not an assistant, and the dataset remains narrow and
synthetic. Judge further progress by unseen held-out results and readable
generation, not just training-step count.


## Run 005: continued language learning on exact same expanded corpus

- **Date:** 2026-11-02 US Eastern / 2026-11-03 UTC
- **Workflow:** https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41558008416
- **Code commit:** d51cfade7ea7d98462bd917aa5da3138c157a65c
- **Previous model:** exact Forge Seed Run 004 checkpoint, same optimizer
  and RNG state; the neural model originated from random initialization.
- **Corpus:** unchanged, content-locked 800 TinyStories train and 200
  validation documents, training SHA256 c4f483c4..., validation SHA256
  263f2b99..., complete manifest 4b59232f...
- **Training:** 4,000 additional AdamW steps, cumulative step **8,000**,
  same 1,875,072-parameter model, batch 4, window 128 bytes, seed 42.
- **Fixed 32-window held-out loss:** **1.410106 → 1.253251** (~11.1% lower).
- **256-window held-out audit:** **1.281533 → 1.109719**.
- **Fixed-window bits per byte at step 8000:** **1.808059**.
- **Validation by global step:** 4000:1.410106; 4400:1.406927;
  4800:1.365598; 5200:1.344570; 5600:1.322622; 6000:1.287775;
  6400:1.276165; 6800:1.273620; 7200:1.246898;
  7600:1.259833 (minor regression); 8000:1.253251.
- **Measured CPU training time:** 216.19 seconds.
- **Generated sample:** starts "Once upon a time, there was a vasiling or
  some as her ways..." with recognizable grammatical scaffolding, but still
  made-up words and disconnected clauses. **Not fluent or conversational.**
- **Artifact ID:** \`13196360153\`, about 20.7 MB, includes public
  training/validation text, optimizer/RNG, manifest, losses and model.
  Expires on **2026-12-03 UTC** unless archived.
- **Next:** preserve this promising checkpoint as an unpublished draft
  Release; compare five generic, fixed-seed prompts with Run 004 before
  making any decision to add compute or change model architecture.

### Promotion decision

**Engineering progress verified.** The trained model has not met the language
quality gate. No Android agent, pretrained teacher, user memory, subjective
personality or autonomy has been added to these training weights. Keep the
model distinct from the historical draft Run 003 and preserve both reference
checkpoints for reproducibility.


## Run 005 follow-up: fixed-prompt generation and durable draft archive

- **Qualitative workflow:** https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41558718634
- **Identical public dataset and architecture:** Run 004 (step 4,000) versus
  Run 005 (step 8,000). Five generic, nonpersonal prompt prefixes; same
  fixed sampling temperature and seeds; no training in the comparison.
- **Observation:** Later samples more often start with familiar story
  structure (e.g. "there was" and "was very excited") but still contain
  grammatical errors, invented words, repetitions, and disconnected
  sentence content. **The model does not pass the human-readability gate.**
- **Checkpoint archive workflow:** https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41558718548
  successfully audited and created GitHub **DRAFT** Release
  \`forge-seed-run005-v0.6\` (not published as a capable model).
  It includes the resumable checkpoint, source notice, model metrics, public
  train/validation samples, dataset manifest, and audit SHA256 checksums.
- **CI issue and correction:** One new qualitative test initially caught
  an RNG side effect during model initialization. The inspector now
  preserves the caller's RNG state around initialization and generation.
  The model-training weights and prior learning results were unaffected.
- **Next evaluation gate:** Pretrain on an expanded, revision-pinned corpus;
  compare a freshly learned tokenizer to the byte tokenizer at equal
  training-compute budgets. Add human-readable sentence and unseen-story
  evaluations, with explicit failure examples. Do not yet connect an
  unready language model to Android observation or personal memory.
