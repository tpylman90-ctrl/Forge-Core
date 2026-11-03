# Forge Core — Master Development Blueprint v1.0

**Status:** approved technical direction, NOT a claim of current AI capabilities.
**North star:** an independent, locally controlled, deeply familiar thinking partner, with its own consistent perspective; understands a user's reasons rather than merely memorizing preferences; asks before every consequential external action.
**Canonical network:** our own transformer initialized from random weights. A pretrained teacher, if separately allowed, is a distinct optional tool—not a shortcut that substitutes pretrained weights for Forge Core.

## 1. Why the approach changes

Forge Seed v0.6 has a verified 1,875,072-parameter scratch-trained network at 8,000 steps on a source-content-locked TinyStories preview. Held-out next-byte prediction improved, but its generated samples still contain nonsense and cannot support reliable conversation. **This is linguistic-pattern learning, not understanding or reasoning.**

The next objective is not simply training longer. It is to learn whether we can teach structural relationships (word order, roles, negation, agreement, cause/effect) in addition to distributional language, and to measure this against an **equal-compute baseline**.

Two senses of *why* must stay separate:

- **Linguistic why:** Why is this sentence grammatical? Who did what to whom? What changes if we swap subjects/objects or introduce negation? These are supervised, testable properties of language.
- **Personal why:** Why did this particular user choose an action? This is not knowable from app switches or text probabilities alone. Forge must ask respectfully and save hypotheses only on confirmation.

A third concern, **why Forge recommended something**, is an evidence/explanation accountability task. Its explanations must report observable evidence, uncertainty and constraints; they must not assert unfounded access to the model's exact internal causes.

## 2. The system, not one enormous network

### A. Foundation model (neural)
Original, random-initialized decoder Transformer; scalable tokenizer/embeddings/context; causal pretraining on licensed general text. It learns probabilities and representations. It is *not* a truth engine.

### B. Linguistic discovery engine (experimental supervised curriculum)
Produces approved and verified **minimal pairs**, contrastive questions and gold structured answers. Example:

- "The fox followed the rabbit." / "The rabbit followed the fox."
- **Both grammatical**; actor/patient roles are exchanged. A student must identify *who followed whom* in each.
- "The fox follows the rabbit." / "The fox follow the rabbit."
- **First grammatical** for standard English subject–verb agreement; second is not.

Other tasks: negation, tense, pronouns/coreference, quantifiers, spatial/temporal relations, event ordering and context dependence. Specific answers are supplied by fixtures, vetted annotation or checked programs—not by the untrained model's own explanations.

An agent may ask *which rule or relationship matters?* but generated answers are treated as predictions, not evidence. Train/eval vocabulary, templates and documents must be separated. Synthetic-rule tasks alone are insufficient evidence of natural language understanding.

### C. Evidence and reasoning service
Separates observations, retrieved facts, inferences, alternatives and uncertainty. Eventually supports making and falsifying hypotheses, counterfactual predictions and causal tests. A plausible narrative does not prove causation; experiments and interventions matter.

### D. Personality and identity
Two independent profiles:
- **Forge identity:** versioned, consistent manner of collaborating; curious, accurate, constructive and capable of disagreement. User may review it.
- **User model:** preferences, goals and *user-confirmed* principles, with provenance and correction history.

Forge may develop recognizably individual conversational traits without inventing desires, pretending to be conscious or treating its own preferences as authority. This is not a static persona pasted into every answer, nor is it achieved just by changing model weights.

### E. Personal episodic and semantic memory
Encrypted, on-device by default, outside GitHub and model checkpoints. Store confirmed reasoning, episodes, provenance, timestamps, confidence, corrections, retention and deletion rules. **Do not equate retrieval with training.** Only separately opted-in and reviewed records may ever be considered for neural fine-tuning.

### F. Curiosity and inquiry
A safe question-selection engine chooses specific, high-value gaps. For personal intent: Observe (only with source-specific consent) → formulate uncertainty → Ask Why → distinguish user's answer from model hypothesis → propose principle → user confirms/corrects → encrypted save → apply in future → solicit outcome → revise or forget.

User controls "answer", "later", "skip", "never ask about this" and pause. Never assume motives from screen activity, and never collect passwords, private third-party content or sensitive activity for training.

### G. Tools, authorization and user interface
Forge can think, rank alternatives, plan and propose. Its **trusted action executor** must deny actions without authenticated, narrowly scoped user approval, with audit/revocation. A model reply or memory entry cannot grant permission. Android observation is a separate opt-in capability; the current observer prototype stays paused until appropriate reliability, privacy and language quality gates.

## 3. Four independent learning operations

1. **Pretraining:** approved text → tokenize → next-token/byte loss → update scratch weights → held-out text evaluation.
2. **Linguistic supervision:** verified contrasts and structured labels → predict roles, acceptability or changed meaning → compare gold answer → optional auxiliary training only after A/B experiments.
3. **Personal learning:** ask user *why* → confirm context-specific principle → encrypted memory → retrieve and apply; **no automatic weight update**.
4. **Optional model consolidation:** curated, explicitly permitted examples → privacy audit / source manifest → offline training → regression tests → reviewed promotion.

Each loop has different evidence and authorization. Failing one never causes the other to rewrite private memory.

## 4. Controlled experiment for the linguistic Why loop (the immediate next research cycle)

### Baseline
Train a fixed scratch Forge Seed on a pinned, larger, diverse and properly licensed text set. Hold tokenizer, architecture, total tokens, wall/compute budget, seed and evaluation constant.

### Intervention
A matched model receives verified linguistic contrast tasks as an auxiliary supervised signal. Avoid giving it additional effective compute unfairly: match total update budgets and compare a text-only mixture control. One training regime can interleave tasks after a stable base; evaluate both staged and interleaved curricula.

### Evaluation
- **Grammar:** distinguish grammatical/ungrammatical pairs; subject–verb agreement on unfamiliar noun/verb forms.
- **Roles:** answer who-did-what-to-whom after switching word order; compare to explicit gold roles.
- **Semantic changes:** negation, tense, temporal ordering, pronouns and quantifier scope; distinguish grammaticality from meaning.
- **Generalization:** unseen vocabulary, templates, constructions and a fully withheld source dataset, separated before training.
- **Language quality:** held-out byte/token loss, fixed-prompt samples, coherent paragraph/story tests and a small blind human review.
- **Regression:** generation fluency, novelty, catastrophic forgetting, invalid outputs and calibration.

**Promotion rule:** auxiliary linguistic training is an *experiment*, not a mandatory training ingredient. Adopt it only if performance improves on held-out structural tasks **without material degradation** in general language evaluation across repeated seeds and datasets. If it fails, revise or remove it.

No invented claim that a grammar explanation directly reveals the network's true internal reasoning. A generated "because" statement is only a candidate hypothesis and requires independently verified supervision.

## 5. Ordered implementation gates

### Gate 0 — Reproducible data foundation (current + next)
Already verified: scratch architecture, optimizer resumes, SHA256-locked bounded preview, public-data draft checkpoint archives, 8,000-step experiment, CI. Next: immutable upstream revision and licensing audit for any larger corpus; document-level dedup and distinct held-out sources; design tokenizer comparison. Do not train on personal data.

### Gate 1 — Basic language (priority 1)
Create a more representative text curriculum; train/compare byte vs trained-from-scratch subword tokenizers at fixed resource budgets. Improve window/context handling; evaluate readable syntax and unseen text. Larger model/config only after measurable benefit. A 1.9M model is a diagnostic; even ~10–30M is far below the capability of a broad modern assistant.

### Gate 2 — Linguistic why (prototype **now**, train when controlled)
Build transparent contrast generator, label schema, fixtures and lexical/template-held-out tests **now**. Once a stable baseline exists, run matched no-aux/aux experiments as above. Support explanation questions only when answers can be checked; do not have Forge grade itself.

### Gate 3 — Conversation and reliable reasoning
On licensed dialogues and instructional examples, train turn-taking, clarification, following instructions, uncertainty, grounding, consistency and multi-turn continuity. Add benchmarked question answering, error correction and evidence-aware reasoning. Do not train a self-styled personality before it can reliably communicate.

### Gate 4 — Distinct Forge personality (design now, evaluate in conversation)
Write an explicit versioned identity profile; test stable collaboration, constructive disagreement and clear uncertainty. Do not encode an unchangeable "clone" of the user. Track identity changes and human feedback separately from private user memory.

### Gate 5 — Personal why and memory (prototype separately, activate after quality/privacy gates)
Local authenticated encrypted storage with schema migration, correction, export and deletion. Start with volunteered conversations; move to opt-in device observation only when permitted and meaningful. Every reason attributed to a user is a hypothesis until confirmed.

### Gate 6 — Authorized initiative / practical application
Forge may identify possible improvements on its own and bring proposals with evidence, assumptions, risks, and expected outcomes. External changes, contacts, app activity and tool calls require explicit scoped consent. Test denial, audit and revocation before integration.

### Gate 7 — Mobile integration and long-term growth
Benchmark Android device RAM, token speed, battery and thermals. Run locally where feasible, use opt-in alternatives only if explicitly agreed. User memories remain transferable across model versions. Larger neural networks and broader real-world skills depend on demonstrated results and realistic compute/resources; this is a research program, not a promise of J.A.R.V.I.S.-level competence.

## 6. Implementation order for the next engineering passes

**Pass A — now:** Commit this blueprint, build tiny **verified contrast-probe data/evaluation fixtures**, add tests for gold answers, disjoint vocabularies and deterministic generation. Do not update model weights from those fixtures yet.

**Pass B:** Obtain/license-review an immutable, richer text corpus. Benchmark a tokenizer trained solely on allowed training data vs byte tokenizer; lock evaluation. Continue scratch pretraining and generate truly held-out samples.

**Pass C:** Add a safe auxiliary linguistic objective adapter, then test matched baseline/control/intervention runs, multiple seeds. Record token budget, FLOP/CPU time, failures and regression. Accept only real gains.

**Pass D:** Add instruction dialogue and reasoning evaluation. Prototype separate Forge identity and personal “ask why” memory under local-user control, using real consent.

**Pass E:** Add proposed-action approvals, trustworthy tool integrations and Android *after* language, safety and privacy gates.

## 7. Compute, data rights and honesty

- GitHub Actions CPU runners work for unit tests, data validation and small experiments, not endless free large-model training. Use bounded, manually approved jobs, checkpoints and explicit retention limits.
- Third-party public datasets are *not automatically* permitted for any training or redistribution. Review the actual license, usage terms, revision and original source; keep attribution and prohibit private material.
- A pretrained model can serve as a **separate optional teacher** to prepare proposed linguistic labels, but those labels need independent checks; no pretrained weights are copied into Forge Seed.
- No unsupervised self-training on model-generated "why" explanations, no broad phone surveillance, no personal data in CI, and no action execution without consent.
- A declining held-out loss is a useful metric; fluent, grounded dialogue is a *different* milestone. Report failed tests and contrary results.

## 8. Five-year success measures (aspirational, not guaranteed)

- Language: coherent multi-turn communication on unseen topics, measured against blinded baselines.
- Structural understanding: correct actor/patient, negation, temporal and grammatical predictions on novel examples—not just memorized templates.
- Evidence: distinguishes facts, assumptions, counterfactual possibilities and uncertainty; corrects itself after feedback.
- Personal understanding: retrieves confirmed reasons accurately, never invents personal motivations, incorporates corrections reliably.
- Independent perspective: asks valuable questions and offers constructive disagreement without flattery or arbitrary opposition.
- Autonomy: proposes useful plans, explains them and **waits for explicit approval** before real-world action.
- Privacy/control: locally owned, portable, encrypted history; verifiable review/correction/deletion and independent security checks.
- Continuity: a model upgrade preserves personal history, identity profile and safeguards, without requiring sensitive information in public code or training.

**Canonical standard:** Do not ask "How many hours has Forge trained?" as the sole metric. Ask "What can this checkpoint do reliably, on new examples, that the previous one could not?"
