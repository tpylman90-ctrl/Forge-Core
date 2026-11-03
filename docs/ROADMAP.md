# Forge Core — Language-first roadmap (v1.0; aligned to the Master Blueprint)

## Canonical blueprint

[Master Blueprint v1.0](MASTER-BLUEPRINT.md) defines the complete system,
the two different meanings of "why", trial design, compute restrictions and
long-term priorities. The initial [Linguistic Why Lab](LINGUISTIC-WHY-LAB.md)
is a separate diagnostic research fixture and **does not modify Forge Seed**.

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
- [x] Verified improvement from step 20 to 1020 on the same content-locked held-out corpus (Run 003); wider audit independently improved
- [x] Verified expanded-corpus scratch initialization versus matched zero-step random-weight baseline (Run 004); text samples remain fragmentary
- [x] Verified Run 003 draft archive `forge-seed-run003-v0.5` after integrity audit; remains unpublished

**Gate:** genuine public-text learning evidence; not just memorizing a tiny demo.

## Linguistic Why mini-gate [prototype now; train only after controlled baseline]

- [x] Original rule-labeled minimal pairs for role reversal, agreement and
  negation; separate hidden-gold eval export; disjoint subject vocabularies
- [x] Deterministic tests reject overlapping actor vocabularies and verify
  labels and public-question separation
- [ ] Add template-, verb- and corpus-disjoint external evaluation
- [x] Implement first **small null-label controlled A/B study**, starting from preserved Run 005 public-data scratch model; run and independent validation pending
- [ ] Repeat on an immutable fully reviewed corpus with matched-compute text-only control, multiple seeds, unseen constructions and external labels
- [x] Three controlled A/B runs completed (original gold labels vs
  shuffled answers, frozen checkpoint), with public-language retention checks
- [x] Run 002/003 public-language rehearsal largely prevented next-byte
  regression; actor/patient role skills still at chance, including expanded
  held-out set (160/320)
- [ ] **NO promotion:** defer neural Why adaptation until language and
  instruction-following gates improve, then replicate with external
  annotation, independent templates and multiple seeds

## Friday November 6 — Forge Seed v0.7 early-language experiment

See [Preschool Readiness Gate](PRESCHOOL-FRIDAY.md). "Preschool" is a
provisional human-language skill rubric, not an automatic label.

- [x] Source revision pinned to exact TinyStories commit; bounded raw
  train/validation separate-file loader implemented
- [x] Scratch 2048-token BPE trainer and revised scratch model experiment
  with tokenizer SHA256 and resumable optimizer, RNG and fixed sample prompts
- [ ] Verify actual GitHub Actions training, document measured sample quality
- [ ] Implement unseen who/what/color and simple instruction assessments
- [ ] Pass the rubric or transparently report failures on Friday
- [ ] Review exact-source retention and licensing before publishing weights

## Milestone L2 — Coherent language and dialogue [NEXT]

- [ ] Move beyond preview rows to immutable, complete, licensed training documents
- [ ] Train a useful Forge Seed with sufficient steps, diverse immutable documents and suitable compute; Run 005 at 8,000 steps remains fragmentary
- [ ] Review byte-level tokenizer inefficiency; benchmark subword tokenizer alternatives
- [x] Five-prompt qualitative inspection Run 005 versus Run 004 completed; recognizable syntax emerging, but grammatical errors and invented words remain
- [x] Human-reviewed initial five story prompts: **fluency NOT achieved**; fixed benchmark and source-preserving archive verified
- [ ] Improve unseen narrative continuity, grammatical consistency and basic QA before claiming language capability
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


## November 3 verified v0.7 and next learning experiment

- [x] Revision-pinned 10 MiB public training/2 MiB public validation prefixes, scratch BPE, original 2.22M Transformer and 14,000 steps
- [x] Unseen mirrored preschool QA baseline **19/40** (~chance), not passed
- [x] First broad verified-label QA study failed to transfer: **20/40**, shuffled **21/40**
- [x] Sixteen-example sanity calibration memorized **16/16 after 100 updates**, so output learning works on repeated examples
- [x] Baseline v0.7 research checkpoint preserved in audited **unpublished draft** GitHub release
- [ ] Staged 16→64→256→512 question teaching and independent transfer evaluation
- [ ] Demonstrate robust role/color/count/location comprehension and simple instructions on novel examples before claiming preschool-like ability

## November 3 — contrastive QA next experiment

- [x] Verify staged learning completed: 98.83% training examples correct, but only 18/40 transfer answers (worse than 20/40 frozen)
- [x] Identify shortcut risk: answer position/wording rather than true subject, recipient, color, location or count relations
- [x] Implement new curriculum of **paired counterfactuals and both A/B choice orders**; test deterministic generation, class balance, and cumulative training
- [ ] Run gold-label versus shuffled control; audit both-mirror transfer, held-out text loss, and generated stories
- [ ] Repeat promising results with independent templates and seeds before any research-candidate promotion

## November 3 — counterfactual and direct answer research

- [x] Paired counterfactual training completed (3,250 steps/arm), with exact-source tests and no promotion
- [x] Verified results: transfer only **23/40** gold vs **22/40** shuffled; both choice orders correct **5/20**; failed quality threshold
- [x] Story loss 2.6961 vs 2.6380 frozen; close but weak comprehension evidence
- [x] Implement bounded **direct-answer text** teacher-forcing research with verified gold vs shuffled answers and original-language rehearsal
- [x] Direct-answer research completed: 256/256 training, 9/20 original, 6/20 unfamiliar, while shuffled control 4/20 and 3/20
- [x] Held-out story loss remained close (2.6616 verified vs 2.6380 original)
- [ ] Larger independent, novel direct-answer test and multiple training seeds
- [ ] Broaden colors/quantities/roles/wording with controlled data compositions, preserve language rehearsal
- [ ] Evaluate authentic single-step instructions and coherence before claiming preschool ability


## November 3 — next broad language-generalization cycle

- [x] 3,200-question compositional direct-answer generator (five skills)
- [x] Separate 50-question progress set and 200-question final challenge
      with disjoint names, objects and surface-form templates
- [x] Original-weight versus gold-text and shuffled-text controls, with
      shared public-language next-token rehearsal
- [ ] Run CI and 2,500-update-per-arm bounded experiment; inspect
      held-out per-skill exact answers and any general language regression
- [ ] If promising, independently replicate on new templates/seeds;
      evaluate free-form one-step instructions, not only exact short answers
- [ ] Preserve source checkpoint integrity and refuse automatic promotion


## Nov 3 compositional result / immediate next step

- [x] 3,200-example compositional direct-word training completed successfully
      (research Actions run 41649770774)
- [x] Verified **49/200** unseen internal synthetic QA vs shuffled
      control **13/200**, source original **0/200**; relation gaps persist
- [x] Independent 50-instruction audit completed:
      **11/50** verified vs **4/50** shuffled, **0/50** original
- [x] Found specific skill deficits: **0/40** giver and recipient
      QA, **0/10** repeat/select/recipient instructions
- [x] Start new copy/list-selection/semantic-role apprenticeship from
      a separate copy of verified *research candidate*; source hashes
      checked, no promotion
- [x] Check fresh 120-instruction held-out test (20/120 verified vs
      19/120 parent and 8/120 shuffled); regression in
      existing 200-QA and story loss
- [ ] Repeat only truly promising results with different seeds and
      independently authored natural language examples


## Nov 3 — Teacher v1 and pointer architecture gates

- [x] Implement optional **local AI teacher curriculum advice**, validated
      strictly from aggregate development scores; no GGUF is bundled
- [x] Implement verified lesson/answer generator with checksum and rule
      regeneration, training-vs-development isolation and adaptive selection
- [x] Run controlled 280-update Forge neural pilot; verified **19/120**
      vs parent **22/120** and shuffled **14/120**; negative result
- [x] Run 73,920-parameter frozen-backbone neural pointer:
      **42/120** development vs **7/120** untrained and **4/120** control
- [x] Run corrected 360-case novel-template exam:
      **41/360** pointer vs **12/360** incorrect-source control; generalization
      drops sharply compared with development, so no promotion
- [ ] Investigate entity-role binding with structured attention instead of
      repeating the same source-pointer or answer-token training templates
- [ ] Diversify sentence grammar, use externally sourced and independently
      verified language questions, and replicate across random seeds
- [ ] Independently assess reading comprehension, one-step instructions,
      sentence coherence and retention before any preschool-capability claim
- [ ] An optional locally supplied licensed GGUF may be connected to
      teacher advice; do not claim it was used during hosted experiments

## Nov 3 — independent frozen-pointer semantic-role diagnostic

- [x] Prespecified three new seeds, four grammar structures, 384 questions
      with source words and independently checked giver/recipient gold
- [x] Audited source/parent/pointer SHA256 lineage and frozen weights
- [x] Verified both Python CI and dedicated no-training experiment in
      Actions run **41680677587**, artifact **13252862393**
- [x] Measured **83/384** correct copied names vs **11/384** wrong-source
      control, but only **9/192** both-correct role questions and
      **11/192** both-correct entity swaps — NO PROMOTION
- [ ] Benchmark role-vs-copy errors with an answer-independent entity
      detector, not a gold-derived candidate filter
- [ ] Design role-conditioned head against matched shuffled-label
      control using entirely separate training/exam grammar families
- [ ] Assess new independently authored natural-language sentences and
      multiple seeds before any capability claim

[Detailed result and next method](POINTER-ROLE-RESULTS.md).
