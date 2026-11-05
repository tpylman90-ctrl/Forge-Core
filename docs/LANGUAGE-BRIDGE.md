# Forge Core language bridge — optional helper, NOT Forge's pretrained foundation

The canonical training plan is to build Forge's own neural language model from **random initialization**. See [original language training loop](LANGUAGE-TRAINING-LOOP.md). The pretrained teacher does not provide Forge Seed's model weights or replace that training trajectory.

## Why two models?

Our randomly initialized 10–30M-parameter language model is a controlled research experiment. It cannot reliably ask natural, context-sensitive questions or understand ordinary conversation without substantial training and evaluation.

To avoid making the user manually label every decision with rigid fields, Forge Core will also support a **separately licensed pretrained model as a temporary language teacher**. This teacher may:
- Ask why and ask useful follow-ups.
- Summarize an explanation as a *tentative* general principle.
- Retrieve relevant user-confirmed principles from the encrypted local memory.
- Propose alternative interpretations while admitting uncertainty.

The teacher does NOT autonomously approve changes to personal memory, issue external actions, or train the scratch model. Its suggestions must be reviewable; user confirmation controls memory writes.

## Initial offline candidate

Qwen3-0.6B, distributed under Apache-2.0, with a separately sourced GGUF quantization (check the exact distributor and license). A commonly available Q4_K_M quantization is about **0.48 GB** on disk, excluding inference RAM and runtime overhead. It may run slowly on budget Android phones. Model suitability, battery consumption, output quality and exact license must be measured, not assumed.

Inference runtime: **llama.cpp** (MIT), with its official Android bindings considered for the standalone Forge Core app. Planned workflow: select a local user-provided GGUF file via Android's file picker, import it into app storage, run inference on-device, and never require internet connectivity after the user obtains the model.

No model file is bundled in the small debug APK; no online model download occurs automatically. During model-loading/inference, show memory and battery expectations and allow cancellation.

## Current Android integration

Forge Core Android now includes an optional, user-imported local GGUF chat
bridge using the pinned official llama.cpp Android sample. The APK contains the
runtime but **not** model weights; the user selects a local `.gguf` file, the
app checks its header and digest, and the user explicitly loads or unloads it.
Its responses are labeled as a separate pretrained helper in the encrypted
journal. They are not Forge Seed output and are not used to train the scratch
model. No internet permission or automatic download is used.

This bridge has not been verified on the user's phone. Actual compatibility,
RAM use, speed, heat, and battery draw depend on the selected GGUF and device.

## First Python code-level integration

The optional Python module \`forge_core.language.local_teacher\` invokes a **local** installed llama.cpp CLI binary with a **local** GGUF file. It is suitable for desktop experiments with pretrained conversational ability and unit-tested command construction. Its tests do not claim actual model quality.

Example after obtaining a locally licensed GGUF and llama-cli:

    python -m forge_core.language.local_teacher --llama-cli /path/to/llama-cli --gguf /path/to/model.gguf --observation "I rejected the second car body"

Questions are proposals, never authoritative observations. Do not automatically feed private memory into a model or externally hosted inference system. User consent and data locality must be enforced by the Android integration.

## Acceptance gates for the mobile language engine

1. Real Android build and installation with arm64 native libraries.
2. Import GGUF via Android Storage Access Framework, validate size and format, no implicit downloads.
3. Generate an English follow-up for a harmless test observation; cancel while generating.
4. Record max RSS, speed (tokens/sec), startup latency and battery/thermal behavior on the actual device.
5. Keep local memory encrypted, never write raw dialogue to logs/CI.
6. Separate model text from trusted instructions and user-confirmed principles.
7. Compare response usefulness against hardcoded templates with user feedback.
