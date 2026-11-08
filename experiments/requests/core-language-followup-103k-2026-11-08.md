# Forge Seed v0.7 103k follow-up

Continue the verified research candidate from run 42291976445, checkpoint SHA-256
ec9a85b7a9d3787955d89d82f37bc2b494d96cc6ab5c106138f10285cdd2a868.

This is a small, actual language training run from Forge's own weights:
- 2,000 maximum updates, validation every 500 updates, patience 2
- learning rate 0.0000125 with optimizer moments and sample RNG resumed
- TinyStories revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64, CDLA-Sharing-1.0
- fixed validation plus a new 12 MiB-offset test range; compare parent and candidate
- archive data hashes, optimizer/checkpoint lineage, evaluation, and capability diagnostics

Research only. Do not promote or integrate into Android. A positive result is a small
measured training gain; a loss is retained and reported as a useful comparison.
