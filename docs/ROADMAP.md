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
- [ ] License-reviewed dataset intake and provenance manifests
- [ ] Training/validation split with held-out loss and perplexity
- [ ] Baseline experiments (dataset, seed, duration, compute, loss curve)
- [ ] Demonstrate controlled overfitting on a tiny licensed sample

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
- [x] GitHub Actions Android debug build workflow created (success must be verified)
- [ ] Successful Android APK build and installation on target phone
- [ ] Real device permission/reliability tests and battery impact
- [ ] Feasibility benchmark on target phone (RAM, tokens/sec, thermal, storage)
- [ ] Android interface, accessibility and opt-in sensors
- [ ] Offline memory and explicit synchronization policy

## Measurement rules

Record model architecture/parameter count, reproducibility seed, train/validation bytes, source/license, hardware, runtime, final/held-out loss and failure examples for every experiment. Separate **verified** from **planned** results. A smoke test passing is not a trained assistant.
