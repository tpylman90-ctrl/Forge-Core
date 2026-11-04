# One-time Forge English v4 — positional character learning

The v3 run 41837615641 demonstrated that our existing scratch-trained
language model can sometimes write an unfamiliar **first** character,
but cannot generally compose unfamiliar full strings.

Authorize one bounded successor research experiment, with no change to
the canonical model or Android code:

- Source: audited original Forge Seed v0.7 from public Actions run
  41572257363; no externally pretrained model weights.
- Independent original SHA256-partitioned 3–7-letter source strings:
  150 training/seed, 28 development, 48 final test.
- Explicitly supervise the letter at every one-based source position,
  from positions 1–2 to all positions 1–7.
- Evaluate single-letter generated correctness at every position,
  **externally assembled** whole-string exact matches, and termination.
  Do not call assembly unaided word generation.
- Controls: ordered vs exactly sample-matched shuffled curriculum,
  within-position shuffled target letters, first-letter-only training,
  and original public-story-only baseline.
- Two research seeds 10331 and 10332; each 4×110 optimizer updates per
  model across five arms; one serialized CPU GitHub runner with a
  **45-minute overall job limit**, no automatic repeat or monitoring.
- Public-story rehearsals preserved for teaching arms; identical
  frozen/adapted 48-window language retention metric.
- Select weights using development metrics only, score the new synthetic
  exam once after checkpoint selection, preserve failed results.
- All weights and optimizer/RNG states research-only; **no promotion**.
  Android signing, mobile app, personal memory and other AI models are
  untouched.
