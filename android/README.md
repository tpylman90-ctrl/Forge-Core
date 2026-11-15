# Forge Core Android v0.4.0 — Local Language Bridge & Release Signing

> **Android release signing:** Version 0.4.0 (code 5) keeps the permanent
> signing workflow using four GitHub Actions repository secrets.
> See [Release signing and upgrade instructions](docs/RELEASE-SIGNING.md).
> The release APK is produced only after the secrets are configured;
> previously signed debug APKs may not update in place, and uninstalling
> erases local encrypted data.


The launcher opens **Chat, History, Memory, Settings, and Diagnostics**. Chat
uses an encrypted on-device journal. An optional **local language bridge** can
reply using a GGUF model selected on the device. Its replies are clearly marked
as a separate pretrained helper; they are not Forge Seed and do not train or
alter Forge Seed weights.

- The previous v0.2 MainActivity and ObserveService remain operational inside
  **Settings > Observation & permission controls**. Observation is optional
  and foreground-visible; manual teaching requires no Usage Access.
  Observation notifications still open the original why-question dialog.
- Confirmed principles use the original EncryptedMemory schema and Keystore
  alias; no reset, migration or automatic conversion of journal notes.
- History and Memory have separately confirmed deletion; neither silently
  modifies the other encrypted store.
- Diagnostics identifies model provenance, app and permission state. No
  INTERNET permission or screen-reading accessibility service is added.
- In **Settings > Local Language Bridge**, choose a `.gguf` file you have
  permission to use. The app checks the GGUF header and SHA-256, copies the
  file into app-private storage, and waits for you to tap **Load**.
- The inference runtime is the pinned upstream llama.cpp Android library under
  MIT; its license notice is included in the APK.
- **Unload model from memory** frees RAM while retaining the imported file.
  **Remove imported model file** frees its app-private storage. Messages remain
  in the encrypted journal and are labeled as teacher replies.
  No message becomes a confirmed principle automatically.
- Native integration details: [Model Adapter Contract](docs/MODEL-ADAPTER.md).
- Application ID remains `ai.forgecore.mobile`. Version 0.4.0 (code 5).

**Install caveat:** GitHub-hosted debug builds may use different temporary
debug signing keys. If Android refuses an in-place update due to a signing
mismatch, uninstalling the installed app erases its encrypted local memories.
There is no automatic backup/export or recovery in this build. Preserve
important memories rather than uninstalling solely for a visual review.

**Verification status:** Source changes do not prove a green Actions build
or working phone experience. Check the actual run and test the APK.

---

# Previous v0.2 Companion — First Breath

This is a **standalone experimental Android app** (package ai.forgecore.mobile), separate from Forge-ai and the Python neural model. It has no internet permission, no screen capture and no content-reading accessibility service.

## What the first prototype can do

- Explicit Android Settings grant of app-usage access (special access, not automatically granted).
- Show recently used package identifiers so the user can **choose** which apps may be observed. The selection is stored encrypted in app-private storage.
- Start a **visible foreground service** with a persistent notification and Stop action. The service queries app-foreground transition events roughly every nine seconds.
- On app switches to a selected app, display an optional *why* question notification. The user can skip or open the app to answer.
- Only after explicitly pressing **Confirm & remember**, save the user's reasoning and proposed general lesson. Confirmed memories are encrypted using AES-GCM with an Android Keystore key.
- Review and delete all confirmed principles from the app.
- Observe nothing while the service is stopped; device lock pauses queries; no raw history is saved.

## Build

GitHub Actions workflow: `.github/workflows/android.yml`. Install the generated **debug** APK for testing only. It is not a Play-distribution build. Local SDK 35 + NDK 29 + CMake 3.31.6 + JDK 17 + Gradle 8.10.2:

    cd android
    bash scripts/setup-llama.sh
    cd ..
    gradle -p android :app:assembleDebug

The workflow must show a successful green run before claiming that an installable APK exists. This README is a specification and source guide, not a claim of device verification.

## Android 16: Forge absent from Usage Access

Android 16 restricts usage-access permission for certain sideloaded APKs. If Forge does not appear in Usage Access settings, open **Android Settings → Apps → Forge Core → three-dot menu → Allow restricted settings** (if shown), then return to **Apps → Special app access → Usage access** and enable Forge Core. The new APK provides an in-app shortcut to its App Info page and troubleshooting instructions. Android and OEM variants differ; users may not see this switch, and Forge cannot bypass Android protections.

**Manual teaching works without Usage Access.** Android permission access only gates observation of selected app-switch signals, not learning through direct conversation.

## Local chat bridge setup

1. Obtain a compatible GGUF from a source you trust and follow its license.
   Forge Core does not download model weights.
2. Open **Settings > Local Language Bridge > Choose local GGUF model** and
   select the file. Import checks its header and digest, then stores it privately.
3. Tap **Load local model** and wait for the ready status. Return to **Chat**;
   helper replies are labeled **LOCAL LANGUAGE BRIDGE · NOT FORGE SEED**.
4. Tap **Unload model from memory** when you finish. To remove its storage file,
   unload it first, then choose **Remove imported model file**.

Model loading can take several minutes and may exceed a phone's available RAM.
The encrypted journal is not replayed as context after an app restart; clearing
history unloads the model first so its in-memory conversation context is removed.

The APK does not ship model weights. The user obtains a compatible GGUF from a source they trust and imports it locally. The scratch-trained Forge Seed remains a separate research model and is not connected as a chat engine. See [language bridge](../docs/LANGUAGE-BRIDGE.md).

## First phone test

1. Install the debug APK and open Forge Core.
2. Grant **Usage Access** in Android Settings and return.
3. Tap **Choose apps** and explicitly select one or more non-sensitive apps from the prior day's observed package names.
4. On Android 13+, grant notifications when prompted. Tap **Start observation** again.
5. Open one of the selected apps; Forge may post a question notification about the switch. Tap it to answer, propose a principle and confirm the memory.
6. Return to Forge to review or delete confirmed memories. Use **Pause / stop**, or the notification's **Stop observing** action, to end the foreground service.

Do not grant observation for banking, health, messaging, authentication or other sensitive apps in this experimental build. Observation permission only reveals app transitions, **not why the app was opened**. Questions remain hypotheses until answered and confirmed.

## Limits and known caveats

- This is **not** neural-generated questioning or continuous screen understanding. It asks a simple template question after a selected app is foregrounded.
- System/device restrictions or battery policies may delay events/notifications, and foreground-service use requires review for app-store distribution.
- Android usage access is a powerful permission even though the implementation filters by a local allowlist. Use it only on a device you own/control.
- Android Keystore-bound encryption protects saved contents at rest, but **this prototype has not received independent security or privacy review**. A compromised/unlocked device or app process may still expose decrypted data.
- Losing the Android Keystore key (app reinstall/clear-storage) means stored encrypted memories cannot be recovered. App backups are disabled.
- The initial UI provides **delete all**; per-principle edit/export/recovery and detailed audit tools are future work.
- App-selection changes take effect on the service's next polling cycle, if running.
- The app works offline; it does not include or silently upload private data to GitHub.
