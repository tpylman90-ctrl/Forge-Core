# Forge Seed dialogue bootstrap CPU replication

**Run:** [GitHub Actions run 44657287501](https://github.com/tpylman90-ctrl/Forge-Core/actions/runs/44657287501)  
**Result artifact:** retained in GitHub Actions through 2026-12-20  
**Source parent:** Forge Seed v0.7, run 42552572467, step 107250  
**Parent checkpoint SHA-256:** `848553a81549c1269c360c5d07c57adde92a7816a0bc19b61ecc8028c93fee99`

## Result

Two additional CPU seeds trained the same correct-target and shuffled-target
arms for 160 updates each. Both started from the same frozen parent and used
the same held-out synthetic exam and TinyStories retention check.

| Seed | Correct-target | Shuffled control | Frozen parent | TinyStories loss change | Gate |
| ---: | ---: | ---: | ---: | ---: | :--- |
| 43, initial pilot | 122/128 (0.9531) | 119/128 (0.9297) | 103/128 (0.8047) | +0.016676 | Pass |
| 53 | 124/128 (0.9688) | 120/128 (0.9375) | 103/128 (0.8047) | +0.016792 | Pass |
| 73 | 124/128 (0.9688) | 119/128 (0.9297) | 103/128 (0.8047) | +0.016222 | Pass |

The correct-target candidate beat both comparisons in all three seeds. Its
margin over the shuffled-target control was 3, 4, and 5 exam items respectively.
All three retention changes were below the pre-set +0.05 limit.

## Interpretation and disposition

This is a repeatable, narrow improvement in choosing a correct response over
an intent-matched distractor on synthetic recall, correction, clarification,
and uncertainty questions. The exam uses 128 examples and deterministic
templates. It does **not** demonstrate fluent conversation, reliable
generalization, or readiness for interactive use.

No checkpoint was promoted or substituted for Forge's source weights. The
candidate weights, per-seed reports, manifests, and aggregate summary remain
in the workflow artifact for review; they are research-only. Any next step
should add less templated, held-out probes and raw-generation review before
considering a promotion gate.

