# Forge Core Android — Model Adapter Contract v2

## Scope and model identity

The app has two strictly separate model identities:

- **Local language bridge:** optional, user-imported pretrained GGUF used for
  conversation while Forge Seed develops. It runs offline and its messages use
  the `teacher` journal role. The app must never call its responses Forge Seed
  output or treat chat messages as training data.
- **Forge Seed:** the independently trained Forge model. It remains unavailable
  to this Android build until a checkpoint, tokenizer, runtime and provenance
  pass the native integration gates below. Its future replies use the `forge`
  journal role.

The user can continue saving encrypted notes when no model is loaded. A missing
or invalid model never creates a generated placeholder response.

## Current implementation

`LocalLanguageBridge.kt` wraps the pinned official llama.cpp Android sample.
The user selects a `.gguf` file through Android's Storage Access Framework;
the app copies it into app-private storage, checks its GGUF header, records its
SHA-256 and waits for an explicit **Load** action. Model weights are not
bundled, downloaded, or uploaded. The native library runs offline.

The bridge system prompt identifies it as a separate pretrained helper. The
encrypted journal records its replies under `teacher`; history shows that label
on every such message. The independent encrypted confirmed-principles vault is
not read or modified by bridge inference.

## Adapter API

- `status()` reports whether inference is genuinely loaded and ready, the local
  model name, bridge version, digest prefix and current operation state.
- `generate(prompt, listener)` streams at most 256 generated tokens from the
  local engine. Inputs are bounded to 5,000 characters.
- `cancel()` cooperatively stops generation. Unfinished output is not written
  into the journal.
- `importModel`, `loadModel` and `unloadModel` are explicit user initiated
  operations. Load and import report failures without deleting prior journal
  entries or confirmed principles.

## Privacy and limits

- The Android manifest keeps no `INTERNET` permission. Inference and imported
  files stay on device; the app does not fetch model weights.
- The user must obtain a compatible model from a source they trust and follow
  that model's license. GGUF header validation does not establish model quality
  or licensing.
- The bridge can consume substantial storage, RAM, battery and time. The app
  supports explicit unload; actual performance needs phone verification.
- The journal is not silently replayed as model context after an app restart.
  The user can inspect the encrypted history without sending it to the model.
- Chat history is encrypted separately from confirmed principles. The app does
  not automatically retrieve personal principles, write memory or train on
  conversations.
- The pretrained helper is not Forge's original neural model. Training and
  evaluation remain in the Forge Core research workstream.

## Forge Seed native integration gates

Forge Seed can replace the unavailable native-model path only after the
training workstream supplies a tested checkpoint, tokenizer and special-token
rules, runtime requirements, memory estimate, generation and cancellation
semantics, output limits, and a digest/provenance manifest. Then verify offline
inference, bounded resources, cancellation and app behavior on real hardware.
Do not modify neural-training code from this Android workstream.
