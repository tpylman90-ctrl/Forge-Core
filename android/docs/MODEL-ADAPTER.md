# Forge Core Android — Native Model Adapter Contract (draft v1)

**Owner:** Android workstream. **Status:** UI-side interface only; no neural inference integrated.
**Implementation:** android/app/src/main/java/ai/forgecore/mobile/ForgeModelAdapter.java

## Purpose

The Android shell must never impersonate the independently trained Forge neural model.
The current adapter is ForgeModelAdapter.Unavailable; its available=false state
means all Chat input is saved as local notes only. No generated response is
fabricated. The model training/reasoning workstream remains independent.

## Java interface

- status() returns ModelStatus(available, name, version, provenance, detail).
  available means inference is genuinely installed, validated and ready.
  provenance must identify Forge-native checkpoint lineage and verification,
  not a pretrained teacher or unnamed replacement.
- generate(String prompt, Listener listener) starts bounded local inference.
  Listener.onToken(String) streams model-generated tokens; onComplete() marks
  completion; onError(String) reports failure. Only one terminal callback.
- cancel() requests cooperative cancellation and stops further token delivery.
- Callbacks may arrive on a worker thread; UI marshals them to the main thread.
  A production adapter must not block the UI thread.

## Integration gates

1. A tested Forge-owned checkpoint, tokenizer, compatible runtime and full
   weights provenance must exist. Research metrics alone do not authorize
   activating the adapter.
2. Inference is offline and runs only locally. The Android package must
   remain without INTERNET permission. No hosted completions.
3. Bound inputs, token output, context, memory, time and resource use. Expose
   failure, exhaustion and cancellation without losing user journal entries.
4. User-controlled confirmed principles and journal entries are separate stores.
   Do not silently ingest, train on, or promote personal history into model
   memory. Request explicit consent for future retrieval or memory writing.
5. No actions are implemented here. Future external actions require a separate
   user-facing proposal, clear scope, explicit approval and verification.
6. Test on hardware: RAM, startup time, battery, heat, token speed, cancellation.
7. Version adapter contract when implementing it. Coordinate through PRs;
   do not edit neural-training code from the Android branch.

## Journal and memory separation

- EncryptedJournal: forge-core-journal.enc, schema 1, dedicated Keystore
  AES-256-GCM alias forge_core_journal_v1, AtomicFile writes, ciphertext cap
  1 MiB, message limit 8000 characters. Roles: user and forge. Only actual,
  completed native-model output should be marked forge.
- Existing EncryptedMemory: forge-core-memory.enc, original schema and alias
  remain unchanged, for observed-app allowlist and confirmed principles.
- Manual teaching still requires the existing Confirm & remember dialog.
  Simply writing into Chat does not confirm a principle.
- Backups are disabled. A differently signed debug APK might need uninstall,
  which deletes local data; avoid destructive uninstall without a safe export.

## On-device UI semantics

Chat: local journal and honest adapter state.
History: encrypted entries with explicit deletion confirmation.
Memory: confirmed principles with explicit deletion confirmation.
Settings: existing optional observation controls and stop action.
Diagnostics: model provenance/availability, storage and permission state;
private texts do not appear in diagnostics.

## Training workstream handoff

When integration-ready, supply checkpoint format, tokenizer/vocabulary and
special-token behavior, Android runtime needs, RAM estimate, token streaming
and cancellation semantics, generation limits, and digest/provenance manifest.
No modifications to src/forge_core/neural/ by the Android workstream.
