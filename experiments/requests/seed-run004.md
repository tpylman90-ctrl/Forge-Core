# Forge Seed Run 004 — One-time new-corpus language experiment

Authorized continuation of the Forge Core language-first program.
This marker triggers one controlled GitHub Actions experiment; no automatic
recurring training is enabled.

- Network: original 1,875,072-parameter seed Transformer, **random weights**.
- Fresh dataset profile: 800 source TinyStories train rows and 200 independent
  validation rows. Distinct from Run 003's exact-content locked 200/50 corpus.
- No inherited weights or personal-memory training.
- Optimizer: 4,000 CPU AdamW steps, batch 4, window 128 bytes, seed 42.
- Validation every 400 steps; 128-window expanded held-out audit.
- Generated text: inspect, do not automatically claim coherent language.
- Save full public sample slices, manifest, license notice and checkpoint
  as a temporary GitHub Actions artifact; **lock exact content hashes after
  success** before any future resumption.
- GitHub Actions job capped at 70 minutes; this is experimental CPU usage,
  not an unattended permanent training service.
