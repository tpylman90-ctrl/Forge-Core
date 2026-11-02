# Forge Core Android Companion — First Breath

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

GitHub Actions workflow: .github/workflows/android.yml. Install the generated **debug** APK for testing only. It is not a Play-distribution build. Local SDK 35 + JDK 17 + Gradle 8.10.2:

    gradle -p android :app:assembleDebug

The workflow must show a successful green run before claiming that an installable APK exists. This README is a specification and source guide, not a claim of device verification.

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
