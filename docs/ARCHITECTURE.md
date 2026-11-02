# Forge Core Architecture v1.0 (research design)

## Separation of concerns

- **Neural core:** Original causal Transformer initialized with random weights; trains on approved public or user-provided non-sensitive corpora.
- **Identity:** Stable principles and evolving, explainable preferences, distinct from user imitation.
- **Private memory:** Encrypted, locally stored observations and user-confirmed principles; separate from checkpoints. **Not implemented in v0.1.**
- **Curiosity:** Structures questions, hypotheses and confirmation before saving principles; v0.1 is an in-memory prototype, not a learning agent.
- **Reasoning/orchestration:** Evaluates options, uncertainties and outcomes; in the future can call models and memory retrieval.
- **Authorization boundary:** External operations require explicit, authenticated, scoped approval through a trusted UI. v0.1 has a deny-by-default *reference policy*, not a production security control.
- **Interface:** Android UI and optional integrations are future independent modules.

## Planned flows

1. User dialog → reasoner → response, with optional **local** retrieval.
2. Observation → uncertainty → ask why → proposed principle → user confirmation → encrypted local memory.
3. Agent proposes tool action → trusted UI authorization → capability-specific executor → audit log → user-visible result.
4. Approved public corpus → license/provenance review → preprocessing → random-initialized neural training → held-out evaluation.

Do not treat in-chat answers, generated text or model outputs as trusted instructions to external tools. Memory retrieval may inform proposals but cannot grant privileges.

## Early-learning curiosity behavior

The user explicitly prefers **frequent questions during the initial learning period**. Forge should begin in **high-curiosity mode**, not a restrained question cadence. Ask while context is fresh when an authorized, non-sensitive observation is novel, the user's motivation is unclear, a decision contradicts a prior expectation, or a result provides an opportunity to learn a deeper principle. Follow-up questions are welcome; avoid generic questions and asking the same one again without new evidence.

- **High (initial default):** Event-driven, frequent purposeful why/how questions. No arbitrary daily cap.
- **Balanced:** More selective questions when both novelty and uncertainty justify interruption.
- **Quiet:** Defer questions to a user-initiated review or session (not yet implemented).
- **Paused:** Do not ask questions or observe new activity until user permission resumes.

The future Android interface needs clear settings, per-app observation controls, a global pause switch, and simple **answer / later / skip / don't ask about this** responses. No screen or app observation without explicit opt-in, even in high mode. Avoid interruptions during unsafe moments, active privacy-sensitive tasks, authentication, or when observation access is absent. High curiosity does **not** mean access to private content, unbounded notifications, or ignoring user boundaries.

The v0.1 `CuriosityEngine.should_ask` method is a **simple in-memory reference policy**. It does not observe Android, send notifications, queue follow-ups, evaluate question quality, or infer an actual user's intent. Those are future tested components.

Track question usefulness (answered, skipped, later, correction, confirmed principle, repeated question, perceived interruption). Let question frequency adapt to what the user finds worthwhile without automatically deciding the user no longer wants to be asked.

## Neural baseline

- Byte-level UTF-8 tokenizer with 256 IDs, reversible for valid UTF-8 text. No downloaded tokenizer weights.
- Decoder-only causal Transformer: pre-LayerNorm, multi-head causal attention, GELU feed-forward, tied embeddings/output head.
- Baseline: 6 layers, 6 heads, hidden width 384, context 256, around 10.8M trainable parameters (count verified at runtime).
- CI-only tiny variant: 1 layer, 4 heads, hidden width 32, context 128.
- Cross-entropy next-byte objective; checkpoint consists of model/config/step only. **No claim of useful linguistic performance.**
- CPU sanity tests; serious dataset training requires compute and careful validation.

## Privacy/security design (future)

Use Android Keystore-bound encryption keys, authenticated encryption (such as AES-256-GCM), private application storage, backup restrictions, key-loss recovery policy, user-visible retention/export/delete controls, and security review. No homegrown cryptography. Do not place real private memories in public Git, CI logs, dataset samples or model checkpoints. Model-weight training on private memories is opt-in and out of scope for v0.1.

## Experiments versus capabilities

| Area | v0.1 | Future |
| --- | --- | --- |
| Original model | Trainable code | Evaluated reasoning/language |
| Personal memory | Design only | Encrypted Android storage |
| Curiosity loop | In-memory draft/confirmation types | Useful calibrated questions |
| Action safety | Deny-by-default example | Trusted, audited mobile consent |
| Identity | Written policy | Consistent, evaluable behavioral profile |
| Android | None | Offline/assisted inference and UI |

## Guiding principle

A model may be small or replaceable; personal memory, authorization, provenance and accountability must remain independently testable.
