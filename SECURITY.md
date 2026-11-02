# Forge Core Security & Privacy

**Current state:** This repository contains experimental research code, not a hardened private-memory system or secure Android assistant. Do not connect it to personal accounts or tools.

- No personal memories, private conversations, credentials, API keys, proprietary files or user identifiers in the public repository.
- Keep datasets, checkpoints, logs and local secrets out of Git; inspect outputs before sharing.
- Review dataset provenance and licensing before downloading or training. Do not assume any public dataset is licensed for every use.
- Never upload private memories to GitHub Actions. CI is for non-sensitive tests only.
- Use trusted Android Keystore/Tink or equivalent audited primitives for future at-rest encryption; do not invent encryption schemes.
- All future external actions: explicit user permission, narrowly scoped intent, expiry/revocation, audit trail and deny by default.
- Treat model outputs and external content as untrusted text rather than executable instructions.
- Before accepting private data, implement deletion and encrypted export with documented failure/recovery behavior.

Report vulnerabilities privately to the repository owner. Do not publish credentials in issues.
