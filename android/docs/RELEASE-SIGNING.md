# Forge Core Android — Permanent Release Signing

This procedure signs `ai.forgecore.mobile` with the project's long-lived
private Android release key. It does **not** affect Android Keystore encrypted
user memory and it does not install a neural language model.

## GitHub Actions secrets (repository level)

In GitHub: **Settings → Secrets and variables → Actions → New repository secret**.
Create all four entries with values from the private owner-held signing packet:

| Secret | Meaning |
| --- | --- |
| `FORGE_ANDROID_KEYSTORE_B64` | Single-line Base64 encoding of the JKS release keystore file |
| `FORGE_ANDROID_STORE_PASSWORD` | Password protecting the JKS keystore |
| `FORGE_ANDROID_KEY_ALIAS` | Signing alias/username inside the JKS |
| `FORGE_ANDROID_KEY_PASSWORD` | Distinct password protecting the signing key |

Copy the literal value only, with no quotes or extra spaces. Base64 is not
encryption: never publish or commit any signing material, including test values.

**Do not regenerate a new keystore for every build.** Keep two offline backups
of the original .jks plus the passwords, protected from unauthorized access.
These signing credentials are not stored in GitHub repository files.

## Workflow behavior

- `.github/workflows/android.yml` still compiles a debug APK on pull requests,
  pushes affecting Android, and workflow dispatch.
- Only on trusted `main` runs (never feature branches or pull requests), if
  all four secrets exist, the same workflow additionally
  decodes the keystore into the runner's temporary directory, compiles
  `:app:assembleRelease`, verifies the APK signature with `apksigner`,
  removes the temporary keystore, then uploads
  `forge-core-signed-release-apk`.
- If none of the four secrets exists, the workflow succeeds with a **debug
  APK only** and gives an explicit status. If partially configured, the
  signing-check step fails rather than silently producing a misleading release.
- Secrets are never available to untrusted PR code in the workflow;
  the signed release step is skipped on pull requests and non-main branches.
- When all four secrets have been created, run **Actions → Forge Core Android
  APK → Run workflow → main** to trigger a signed release artifact.
- Only a successful run with the `forge-core-signed-release-apk` artifact
  is proof that the release was signed. A passing debug build is not.

## Upgrade identity and data preservation

The project application ID remains `ai.forgecore.mobile`.
Version 0.3.1 uses versionCode 4 and creates the first long-lived release
signing identity. A GitHub-hosted **debug** APK has a different signing key:
Android will typically reject installing a release-signed APK over it.
Uninstalling deletes app-private encrypted memories and journal entries.
**Do not uninstall while personal data is present without an approved export
and backup mechanism.** This app does not yet provide one.

Once the permanent signed release has been installed, every subsequent
release should use the **same keystore, alias, and passwords** and increment
versionCode. Private on-device memory may then persist across normal in-place
upgrades, provided the app's key material and storage remain intact.

## Security checks

- The public repository must never include `.jks`, `.keystore`, encoded
  keystores, passwords, or signing keys.
- GitHub Actions secrets are injected only into the signing configuration
  check and signed release build steps; release signing material is deleted
  from the runner's temporary directory after the build step.
- `apksigner verify --verbose` must succeed on the artifact produced.
- There is no Play Store signing enrollment, Play distribution setup or
  independent security audit implied by a successful signing build.
