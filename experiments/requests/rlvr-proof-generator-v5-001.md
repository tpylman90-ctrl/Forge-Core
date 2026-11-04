# One-time Forge RLVR v5 proof-action generation pilot

Scope: the public Forge-Core repository's independent neural research
track only. The Android conversation owns `android/` and its APK workflow.

- Replace the old 2-candidate selection task with explicit formal
  **action generation**: ADD_ONE, SUB_ONE, DOUBLE, HALVE_EXACT,
  APPLY_RULE_0…3 (symbol indices reused per domain) or ABSTAIN.
- Verify every proposed action using the exact bounded Python math/
  Horn-logic interpreter; award complete proof and correct refusal
  only with independent formal checking.
- Three arms, same scratch initialization/optimizer/task schedule:
  outcome RLVR, dense-step RLVR, and dense RLVR with additional
  oracle-generated process labels.
- Two independent seeds **5321, 5322**. **90 optimization updates**
  per arm per seed. One serialized Ubuntu hosted CPU runner with
  a **25-minute job timeout**, no continuous schedule or repeat.
- Balanced feasible/impossible problems, exact-planner and always-stop
  baselines, development-only checkpoint selection and separately
  shifted synthetic final cases.
- Report verified complete proofs, justified abstentions, invalid
  proposals, verifier calls, and domain breakdowns, as well as
  accuracy on unfamiliar arithmetic magnitudes and logic premises.
- Archive only research metrics and temporary experimental checkpoints.
  **No weights may be promoted** to Forge Seed; original language
  weights, Android source and personal encrypted memory remain
  unchanged.

The aim is validated construction of formal reasoning steps—not
simply increasing GitHub runner time or claiming LLM chain of thought.
