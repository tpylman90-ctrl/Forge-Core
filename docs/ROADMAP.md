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

## Nov 3 — question-conditioned neural entity-slot experiment

- [x] Frozen scratch-model lineage, separate two-person candidate extraction,
      and a newly initialized question-conditioned role head
- [x] Disjoint original synthetic training/development/exam grammar, names,
      objects, questions, source parser checks and counterfactual pairs
- [x] Correctly shuffled within question role ACROSS grammar templates;
      matched gold vs null update count and initialization; test control
- [x] Full CI success and experiment run **41712121652** with research
      artifact **13268059936** (no weight promotion)
- [x] Measured **162/288** gold versus **148/288** shuffled and **144/288**
      untrained; both correct giver/recipient questions only **20/144**
- [ ] Isolate whether question wording changes the answer using a
      query-ablated matched baseline and paired-question contrastive training
- [ ] Prepare a NEW independently sourced/annotated final challenge;
      do not reuse the now-examined 288 cases as untouched evidence
- [ ] Replicate promising results across multiple initialization seeds,
      new syntax sources and language-retention benchmarks before promotion

[Detailed independent research report](ROLE-CONDITIONED-RESULTS.md).

## Nov 3 — completed 36-minute contrastive study and factorized v3 pilot

- [x] Verified single CPU 36-minute run **41728138301**, artifact
      **13278198241**, original Forge language backbone frozen
- [x] Three independently initialized control sets; **84,221** rounds
      per arm over **35m 58s** at ~39.1 rounds/s
- [x] Reported **95/540 (17.6%)** both-correct pairs for contrastive,
      **81/540 (15.0%)** cross-entropy, **63/540 (11.7%)** shuffled,
      **0/540** question ablation; *no promotion*
- [x] Preserved [verified long-run report](ROLE-CONTRAST-V2-RESULTS.md)
      before expiring GitHub artifacts
- [x] Implemented original-weight-frozen two-stage question-intent and
      event-giver classifier, with controlled synthetic augmentation
- [x] Full CI and two-seed v3 research run **41743517853** passed;
      research artifact **13281429407**
- [x] Verified story giver labels at **204/288** and **246/288**,
      but question-intent labels only **194/288** and **146/288**
      across different initializations
- [x] Paired v3 factorized **43/288**, wrong-story control **47/288**,
      monolithic comparator **56/288** — **no checkpoint promotion**
- [ ] Build a question-only, balanced paraphrase benchmark with
      **independently authored** sentences and held-out language families
- [ ] Study question-intent accuracy and negative examples where no
      giver/recipient inference is possible
- [ ] Compare contrastive question supervision to a plain classifier
      with matched updates and independent seeds, sealed new exam
- [ ] Attach a question encoder to the story-role head only after
      evidence of question transfer across unseen wording
- [ ] Preserve frozen language-model validation-loss and original
      source/checkpoint provenance when considering any model fusion
- [ ] Implement durable checkpoint retention beyond ephemeral Actions
      artifacts, with no personal memory in public repositories

[Detailed v3 research results](ROLE-FACTORIZED-V3-RESULTS.md).

## Nov 3 — RLVR and formal process-verifier research

- [x] Implement pure arithmetic and six-proposition implication
      verifiers for explicit two-step candidate-state proof traces
- [x] Implement genuine on-policy grouped-baseline REINFORCE
      from randomly initialized weights, with outcome-only, process
      reward, and process-supervised branches
- [x] Include independent formal oracle step-check labels and
      learned process-validity head, with actor-step CE distillation
- [x] Verify correct-final-with-invalid-steps **fails** full-proof metric,
      test controlled CPU budgets, no arbitrary code execution
- [x] Pass full Python CI and GitHub research runs **41744993637**
      and **41745560943**; record durable report
- [x] Observe negative transfer: process-supervised **45/200**
      full proofs vs final-only RLVR **51/200** after 120 updates;
      process validity classification **443/800**
- [x] Refuse checkpoint promotion, leave original neural language
      model weights and private memory unchanged
- [ ] Audit in-distribution versus arithmetic-range and
      symbolic-premise shifts using NEW held-out samples
- [ ] Test independent confidence calibration and verifier
      abstention when neither proposed step is legal
- [ ] Separate the learned step-checker from symbolic oracle
      and evaluate adversarial/counterfactual proof chains
- [ ] Consider text-to-symbolic proof interface only after
      benchmark gains and language retention gates pass

[Detailed verifier learning results](RLVR-PROCESS-RESULTS.md).

## Nov 3 — Verifiable rejection and step-balanced proof learning

- [x] Original scratch-started RLVR policy supports **REJECT BOTH**
      when neither candidate is a valid formal reasoning step
- [x] Independently verify 0-or-1 valid candidates in both arithmetic
      and finite propositional logic, with balanced VV/VN/NV/NN offers
- [x] Test against exact-verifier **240/240** reference and
      always-reject **120/240** baseline, across two independent seeds
- [x] GitHub v3 run **41747102937**, artifact **13283828294**:
      all learned policies collapsed to **always reject**; 0/60
      feasible proofs per seed — NO PROMOTION
- [x] Identify class imbalance: 75% raw validity accuracy equals
      the **always-invalid** majority-class baseline
- [x] Build v4 offline process-learning curriculum exposing both
      steps through formally verified canonical prefixes
- [x] Compare 3:1 positive-balanced candidate BCE, unweighted BCE
      and supervised three-action CE; development-only thresholds
- [x] GitHub v4 run **41747933263**, artifact **13284074002**:
      no arm consistently exceeds 50% shifted balanced validity
      accuracy; some coverage gains produced unsafe invalid accepts
- [x] Publish permanent [v3 results](RLVR-ABSTENTION-V3-RESULTS.md)
      and [v4 results](RLVR-STEP-VERIFIER-V4-RESULTS.md) before
      temporary artifact expiration
- [x] Prototype a **constrained proof-action proposal model** that
      emits real symbolic operation tokens and verifies every state
      transition without supplied candidate answers
- [ ] Measure verified *constructed* proofs, oracle calls,
      false acceptances, and abstention on new operation families
- [ ] Keep symbolic proof-checking authoritative and independent
      of any neural process model or natural-language explanation
- [ ] Only later connect to the random-initialized Forge language
      network, after separate transfer and retention gates

**Decision:** no promotion and no further 36-minute repeats of the
same failed candidate-ranking architecture. The engineering
infrastructure is validated; genuine neural proof reasoning is not.

## Nov 3 — Formal action generation and shortest-depth curriculum (v5–v7)

- [x] Create a random-initialized five-action neural proof proposer,
      explicit arithmetic/Horn-logic executor and exact bounded
      reachability oracle. No raw hidden model CoT.
- [x] CI and one-off v5 GitHub run **41769105341** passed:
      all three RLVR/process arms abstained on 240/240 impossible
      examples but solved **0/240** feasible cases across two seeds.
      [Verified v5 report](RLVR-PROOF-GENERATOR-V5-RESULTS.md).
- [x] Add fully verified, teacher-forced canonical proof trace
      supervision for second and third stages of an operation sequence.
- [x] CI and v6 run **41769831575** passed:
      complete trace imitation learned **22–29%** correct non-STOP
      actions on teacher prefixes but only **1/240** full feasible
      proofs across seeds. [Verified v6 report](RLVR-VERIFIED-TRACES-V6-RESULTS.md).
- [x] Fix depth mislabeling: formally require **shortest** action
      sequences for 1-, 2- and 3-step curriculum lessons.
- [x] CI and v7 run **41770908240** passed:
      900 matched updates/arm/seed in **78.1 seconds**. Staged
      1-step shifted success **93/240** vs flat control **24/240**;
      staged 3-step success **2/240** vs flat **10/240**.
      [Verified v7 report](RLVR-DEPTH-CURRICULUM-V7-RESULTS.md).
- [x] Research-only checkpoints and audit data retained temporarily.
      **No** experimental neural policy promoted to canonical Forge Seed,
      Android, or personal memory.
- [ ] Balance late-stage short-proof rehearsal with sufficient
      **3-step examples**, and compare matched exposure counts to
      isolate genuine compositional transfer.
- [ ] Evaluate on externally authored new operation families and
      longer formal proofs, not repeatedly on the same synthetic
      procedural generator.
- [ ] Consider an auditable, bounded **propose → exact verify →
      retry** execution interface, measuring verifier call cost and
      errors against plain symbolic search before attributing gains
      to the neural policy.
- [ ] Keep language-model training separately reproducible, and
      only later evaluate text-to-formal transfer with language
      retention and user authorization gates.

**Decision:** The original, scratch-started Forge language model and
Android companion can develop independently. These experiments
show an early improvement in single formal operations, **not**
competent three-step reasoning.

## Nov 3 — English Class curriculum for original Forge Seed (measured)

- [x] Build six original school-style teaching levels:
      uppercase alphabet, vowels/consonants, word-part joining,
      noun/verb, sentence subject, and subject–verb–object order
- [x] Generate **independently verified A/B labels** and balance
      correct-answer positions for every skill
- [x] Separate synthetic training, development and final question
      wording and lexical pools; test semantic role facts from
      source syntax rather than trusting a generated label
- [x] Compare ordered lessons against **exactly matched example
      exposure in mixed order**, plus a corrupted-label null
- [x] Source-verify and adapt only the original scratch-trained
      Forge Seed v0.7 checkpoint, with matched public-text rehearsal
      and no changes to Android or user memory
- [x] First successful full CPU run **41775152111**:
      original **48/96**, ordered **48/96**, mixed **53/96**,
      corrupted **47/96** on synthetic final A/B exercises
- [x] CI-verified source/model SHA256 read-only audit
      **41775982211**, artifacts **13297475570**, rescored all
      frozen/adapted models on the same **48 validation windows**
- [x] Confirm mixed model's story-language validation loss ratio
      **1.01182**, within preliminary 5% retention bound, but
      grammar categories mixed: 12/16 noun/verb and word order,
      only **4/16** word joining. **No checkpoint promotion**
- [ ] Introduce generative **short-answer** spelling and word
      composition rather than only two-choice recognition
- [ ] Add proper noun pluralization, subject–verb agreement,
      past/present tense, negation and pronouns as explicit
      independently verifiable classroom lessons
- [ ] Build **human-authored, unseen** sentence and spelling
      evaluation tasks. Do not reuse the now-inspected v1 exam
- [ ] Test reading comprehension and multi-sentence reference
      resolution, including who acted and who received
- [ ] Replicate across independent initialization/data seeds and
      compare to a matched public-text-only adaptation control
- [ ] Monitor held-out public-story loss and free-text generation
      before proposing any language checkpoint promotion
- [ ] Avoid interpreting text-only vowel/capital exercises as
      sound-to-letter instruction: auditory phonics needs separate
      licensed pronunciation/audio evidence and a new model modality

[Verified English Class research and corrected retention metrics](ENGLISH-CLASS-V1-RESULTS.md).

**Decision:** Continue structured English instruction, but redesign
exercises to test **production**, not just A/B answer preferences.
The synthetic curriculum is a teaching scaffold, not evidence of
preschool language competence or independent reasoning.

## Nov 4 — Grounded alphabet copying and counterfactual reading

- [x] Run original-model v2 generative English instruction:
      **0/112** held-out full generated words/sentences for
      every adapted condition. Archived real training outputs show
      memorized practice words rather than unfamiliar source copying.
      [Results](ENGLISH-PRODUCTION-V2-RESULTS.md)
- [x] Teach exact one-letter, spaced letter and joined-letter answers
      with v3. Correct teaching yields **19/192** staged and
      **21/192** mixed exact unfamiliar exam responses; shuffled
      and public-text-only controls **0/192**. Almost all correct
      responses involve a **single character**.
      [Results](ENGLISH-CHAR-COPY-V3-RESULTS.md)
- [x] Test position-indexed letter retrieval with v4:
      **214/498** verified-mixed individual letters versus
      **67/498** shuffled; only **3/96** externally assembled
      complete strings. Identify synthetic position/vowel parity
      shortcut: **84/96** position-2 letters correct, only
      **89/306** at positions 3–7.
      [Verified results](ENGLISH-POSITION-COPY-V4-RESULTS.md)
- [x] Implement v5 independent uniformly random a–z 4–8-letter
      source strings, exact one-character-changed counterfactual
      twins and held-out hash-partitioned source families.
      [Protocol](ENGLISH-UNBIASED-COPY-V5.md)
- [x] Implement four equal-update controls: verified twin pairing,
      identical examples unpaired, position-shuffled answers,
      and source-masked questions.
- [x] Pass v5 CI, merge to main and launch one bounded 900-update
      two-seed original language experiment
      [Workflow](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/41886375209).
- [ ] Report v5 **both-twin-correct**, indices 3–8, newline and
      externally assembled exactness per seed/arm and upstream
      TinyStories validation loss. Do not select based on exam.
- [ ] If unbiased copying fails, run a separate controlled
      pointer/positional encoding architecture test **from scratch**,
      not more unexamined copies of the same flawed curriculum.
- [ ] After actual multi-position source copying improves,
      test unaided novel word spelling and source-variable copying
      into sentences with a genuinely independent exam.
- [ ] Only advance to complex English instruction after demonstrating
      both retained real-text language modeling and new lexical
      source-binding skills.

**No capability promotion:** Explicit position-indexed character
selection is not English reading, phonics, understanding, spontaneous
word writing or human preschool readiness. The original language
checkpoint and the independently developed Android companion remain
isolated.
