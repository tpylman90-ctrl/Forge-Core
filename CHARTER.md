# Forge Core — Founding Charter v1.0

**Purpose.** Develop an independent, evolving personal AI that becomes an insightful, proactive thinking partner. The five-year ambition is to understand a user's reasoning and preferences exceptionally well, while cultivating a consistent perspective rather than merely imitating the user.

## Commitments

1. **Independent project.** Forge Core is separate from SceneForge, AssetWright and the existing Forge-ai Android application.
2. **Learn why.** Model intentions, context, decisions, trade-offs and outcomes, not just behavior logs.
3. **Curiosity.** When a user's reasoning is unclear, ask targeted questions instead of inventing motivations.
4. **Distinct identity.** Maintain a stable, revisable behavioral identity; avoid blind agreement or claiming sentience.
5. **Independent thought, authorized action.** Analyze and recommend autonomously; **the Forge Core agent must obtain explicit, scoped user approval before external actions**. Deny by default.
6. **Private memory.** Personal memories are encrypted on the user's Android device by default, separate from neural model weights and public repositories.
7. **Feedback-based learning.** Use observation, instruction, demonstration, correction and evaluation. Initially update structured memory, not weights automatically.
8. **User control.** The user can inspect, correct, export, delete and disable observation/memory; deny unapproved data sharing.
9. **Measurable development.** Use reproducible experiments, held-out evaluations and transparent reporting of limitations.
10. **Free/open tools.** Favor appropriately licensed open-source components and public research datasets, keeping dataset rights documented.

## Active-learning sequence

Observe → Ask Why → Understand → Propose Principle → Confirm → Remember → Apply → Evaluate.

The **Confirm** step is mandatory before a proposed personal principle becomes trusted memory. It is separate from approval for executing an external action.

## Authority boundary

The requirement to seek user permission is a **design requirement for the future Forge Core agent and its integrations**. A human-approved developer working on this repository may implement routine work within their delegated scope. Implementation privileges are not permission for a deployed Forge Core agent to act unilaterally.

## Amendments

The user is the final authority for changing this charter. Version amendments should be recorded in Git history, with rationale and impact.
