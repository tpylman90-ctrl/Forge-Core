# Retry three-lane Forge training with verified selected checkpoint

Use the primary selected checkpoint file `language/checkpoint.pt` from run 42552572467, verify its SHA-256 `848553a81549c1269c360c5d07c57adde92a7816a0bc19b61ecc8028c93fee99` and step 107250, then run all three independent 10,800-second CPU lanes. The first attempt stopped before training because the recovery step selected the non-primary `best-checkpoint.pt`; no weights were updated in that attempt.
