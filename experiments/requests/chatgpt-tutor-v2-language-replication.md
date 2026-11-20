# Run an independent language tutor-v2 replication

Run `.github/workflows/chatgpt-tutor-v2-language-replication.yml`.

The first v2 language jobs restored the exact parent RNG state, so seeds 43 and 44 were identical and do not count as independent replication. This run uses the same 107,250-step parent weights and optimizer but a seed-specific token-sampling/RNG stream for seed 44. Preserve all lineage and evaluate the tutor-v2, direct, Ask-Why v1, lane-test, and shared-language measures. Keep the candidate research-only.
