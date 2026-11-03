# Forge Core — Roadmap

## Milestone M0: Governance and scaffolding
- [x] Independent public repository
- [x] Charter and architecture documents
- [x] Basic security rules and data exclusions
- [x] CPU CI tests

## Milestone M1: Trainability, not intelligence
- [x] Random-initialized causal Transformer implementation
- [x] Byte tokenizer and supervised next-byte training loop
- [x] Tiny deterministic smoke tests and checkpoint output
- [x] Bounded TinyStories/demo intake, documented license and provenance manifests (not full curated training yet)
- [x] Separate train/validation sources, deterministic held-out next-byte loss and bits-per-byte; full-document split/leakage audit pending
- [ ] Benchmark first real dataset run; record loss curves, hardware and duration
- [ ] Establish measured language-learning progress against a fixed baseline

- [x] Checkpoint includes optimizer, random state, data hashes and step; explicit resume across GitHub runs
- [ ] Durable checkpoint retention beyond temporary GitHub artifacts

## Milestone M2: Memory and active curiosity
- [x] In-memory proposal/confirmation data types
- [ ] Threat model and encrypted Android storage prototype
- [ ] Inspect/correct/export/delete workflows
- [ ] Ask-why selection criteria and user-confirmed principles
- [ ] Relevance/reliability evaluation for retrieval

## Milestone M3: Independent judgment
- [ ] Evidence/calibration tests and disagreement quality
- [ ] Rule/versioned identity profile, avoid sycophancy
- [ ] Permission-bound external tools with audited approvals

## Milestone M4: Android integration
- [x] Standalone Android source scaffold for opt-in allowlisted app-switch prompts, Keystore-encrypted confirmed memory and visible foreground service
- [x] Android debug APK compiled in GitHub Actions (run 41545378574), artifact 13188462347; Python tests passed (run 41545378489)
- [ ] Install and exercise first Android APK on target phone
- [ ] Real device permission/reliability tests and battery impact
- [ ] Feasibility benchmark on target phone (RAM, tokens/sec, thermal, storage)
- [ ] Android interface, accessibility and opt-in sensors
- [ ] Offline memory and explicit synchronization policy

## Measurement rules

Record model architecture/parameter count, reproducibility seed, train/validation bytes, source/license, hardware, runtime, final/held-out loss and failure examples for every experiment. Separate **verified** from **planned** results. A smoke test passing is not a trained assistant.
