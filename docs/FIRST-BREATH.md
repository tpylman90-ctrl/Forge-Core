# Forge Core — First Breath (v0.2)

This milestone has **two honest components**, not an AI that already understands you:

1. **Neural experiment:** An original, randomly initialized byte-level Transformer and local-training command. Its capacity, training data quality and compute determine how much language it can learn.
2. **Interactive teacher:** A command-line question-and-answer loop. You supply an observation; Forge asks a *why* question, you explain your reasoning, propose an explicit principle, and **confirm** before it is stored in an encrypted local vault. This questioning is currently template-driven, not AI-generated.

## Run the interactive prototype

\`\`\`bash
python -m pip install -e .
python -m forge_core.cli demo
python -m forge_core.cli learn
python -m forge_core.cli review
python -m forge_core.cli forget --id YOUR-MEMORY-ID
\`\`\`

The interactive prototype works on a personal computer with Python. It does **not** observe apps or your phone yet; that requires a separate opt-in Android module. It does **not** automatically update model weights from your answers. The vault is AES-256-GCM encrypted using a scrypt-derived key from your passphrase, and only the local encrypted file is saved. This is a prototype, not a completed security audit. There is no forgotten-passphrase recovery yet.

The default location is \`~/.forge-core/memory.vault\`, which must **never** be committed to Git. Passphrases are not saved. Use a strong passphrase and keep a protected backup if needed. Do not place sensitive real-world personal data into experimental builds before a security review.

## High curiosity, not high surveillance

The planned Android Forge observer uses **explicit per-source opt-in**, minimal signals (e.g. app change, user-given event) and user-controlled pause/exclusions. Forge should begin in high-curiosity mode, without arbitrary low question counts. Only useful questions are surfaced; do not interrogate during sensitive contexts or repeat answered questions blindly. Questions are not permission to use personal accounts or execute external actions.

## Next practical Android milestone

- A standalone Android companion app, separate from Forge-ai, with a private local event journal.
- A transparent observation-permission onboarding interface, no hidden monitoring.
- On-device, event-driven question generation; ability to answer/skip/defer.
- Android Keystore-bound encrypted memory, local inspection and deletion.
- An independent action approval surface (default deny), isolated from model output.
- Android build and phone review before enabling broader observation.
