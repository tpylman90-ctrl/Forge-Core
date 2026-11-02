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
