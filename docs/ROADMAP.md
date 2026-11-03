# Forge Core — Language-first roadmap (v0.4)

## Governing rule

**Build and measure Forge's original language intelligence before expanding observer
or assistant features.** Forge's five-year identity, privacy and authorization
charter remains binding throughout, but implementation follows evidence gates.
The existing Android observer and pretrained teacher are preserved as separate,
non-canonical prototypes. Do not expand those features during the language
foundation phase. Forge Core is not SceneForge or Forge-ai.

## Milestone L0 — Foundation integrity [implemented; continue auditing]

- [x] Public, separate Forge-Core repository and charter
- [x] Original random-initialized causal Transformer, not pretrained Qwen weights
- [x] Small smoke tests and reproducible optimizer/checkpoint resume
- [x] Bounded approved data intake, per-document identifiers, split SHA256 and
      exact/near-duplicate screening, with source/license manifest
- [x] Refuse training with a mismatched manifest or overlapping document IDs/hashes
- [ ] Pin and verify an immutable TinyStories source revision before large-scale training
- [ ] Full-document and cross-corpus leakage review and authoritative license assessment

**Gate:** every dataset used for a report has a provenance record, distinct
held-out documents and reproducible hashes. A passing smoke test is not language.

## Milestone L1 — Learn from real text [NOW]

- [x] Three-stage trainability ladder:
  * **tiny:** ~25k parameters (test control only)
  * **seed:** ~1.9M parameters (CPU-feasible language-learning probe)
  * **base:** ~10.8M parameters (original target architecture; not yet substantively trained)
- [x] Fixed, spread-out held-out evaluation windows, checkpoint metadata and
      optional sampled generation
- [x] GitHub Actions real-text experiment with modest bounded CPU steps, artifacts,
      prior-checkpoint resume and explicit source/hash checks
- [x] **Verified:** first TinyStories preview run; 1,875,072 params, 20 steps, held-out loss 5.5328 → 3.5817; see experiment ledger
- [ ] Show repeatable held-out improvements over more than one checkpoint
- [ ] Compare a scratch run to a matched random-weight baseline and record text samples
- [ ] Select durable checkpoint storage beyond 30-day CI artifacts

**Gate:** genuine public-text learning evidence; not just memorizing a tiny demo.

## Milestone L2 — Coherent language and dialogue [NEXT]

- [ ] Move beyond preview rows to immutable, complete, licensed training documents
- [ ] Train a useful Forge Seed with sufficient steps, data, and suitable GPU compute
- [ ] Review byte-level tokenizer inefficiency; benchmark subword tokenizer alternatives
- [ ] Quantify generated text coherence, held-out language behavior and simple QA
- [ ] Add instruction/conversation finetuning after basic language ability is demonstrated
- [ ] Avoid claiming general reasoning until independently tested

**Gate:** intelligible multi-turn language, evaluated against fixed prompts and
unseen held-out text. Model-size increases alone do not satisfy the gate.

## Milestone L3 — Independent thinking identity

- [ ] Version a distinct Forge personality profile, separate from user imitation
- [ ] Evaluate grounded agreement/disagreement, uncertainty and reasoning quality
- [ ] Explain proposed actions and require explicit approval before external execution

## Milestone L4 — Learn why and retain private experience

- [ ] Integrate conversational "ask why" with real language ability
- [ ] Support proposed, confirmed, corrected and rejected principles
- [ ] Keep personal encrypted local memory separate from weights, with export/deletion
- [ ] Evaluate whether applying a remembered principle was helpful

## Milestone L5 — Android and optional integrations

- [x] Experimental opt-in Android observation prototype, preserved but **paused**
- [ ] Device privacy/security and permission testing; user-controlled opt-in
- [ ] Android inference benchmarks and integration after L2 language quality gate
- [ ] No unapproved actions or hidden observation

## Evidence discipline

Track code SHA, source/license, document IDs/hashes, splits, parameters, tokenizer,
training steps, optimizer state, seed, hardware, time, validation loss/bits per
byte, generation samples, regressions and checkpoint retention. Label every claim
as **verified, implemented but unverified, or proposed**.
