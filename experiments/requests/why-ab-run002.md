# Linguistic Why Lab Run 002 — Public-language rehearsal A/B test

One-time, bounded, authorized Forge Core research experiment. No schedule.

First study Run 001 found 19/30 gold-task held-out answers correct, versus
16/30 shuffled-labels and 15/30 frozen; semantic role questions stayed 50%
and public-language held-out next-byte loss worsened (1.082579 → 1.140008).

**Hypothesis:** Adding the same public-language rehearsal minibatch to BOTH
teaching arms may reduce that language regression while preserving any
structural learning. This is a research question, not a promised result.

- Source: same audited, public-only Forge Seed Run 005, ID 41558008416.
- Two independent model copies start from identical original scratch weights.
- Same 120 steps, batch 2, seed 1234, prompt ordering and AdamW settings
  as Run 001. Each step includes the same public-data minibatch for both arms.
- Per-step loss mixes 40% ordinary public-language next-byte prediction
  and 60% verified or within-skill-shuffled A/B language-structure labels.
- Train corpus hash matches source checkpoint. Validation uses same public
  held-out source; human personal memory or private data never enters.
- Preserve full results and clearly mark all adapted weights EXPERIMENTAL.
- Do not promote, publicly release as an assistant, or deploy in Android.
