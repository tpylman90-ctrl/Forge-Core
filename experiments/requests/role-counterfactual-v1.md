# One-time paired giver/recipient counterfactual run

Run one bounded, research-only direct-answer training study using the
successful 28,000-step, 50 MiB public TinyStories scratch checkpoint from
workflow run 42085634121.

- Preserve the frozen source checkpoint and tokenizer unchanged.
- Teach paired transfer events: ask who gave and who received, then reverse
  the transfer direction while keeping the participants and object fixed.
- Train 3,200 short-answer examples grouped into four-row counterfactual
  episodes, with the preregistered 160 / 800 / 3,200 curriculum.
- Compare verified answers against within-role shuffled answers, using the
  same event batches, optimizer schedule and public-language rehearsal.
- Keep the 200-row final set hidden until all training is complete. It uses
  different names, objects and wording, and separately reports 100 rows
  with verbs absent from training.
- Report greedy generated exact-name accuracy overall, by role, by verb
  slice and by four-answer episode; also measure held-out public language
  loss and preserve fixed story samples.
- Artifact contains research-only candidate weights and metrics. No model
  promotion, Android integration, personal data, or external actions.
- Run integrity tests before training; stop and report if any check fails.

One seed is a diagnostic run, not a capability certification. If the result
shows transfer above both the frozen source and shuffled control, replicate
on new examples and seeds before considering any promotion.


Retry note: the first workflow attempt stopped during bounded input validation
because the workflow expressions were escaped. No training occurred. The
workflow input expressions are corrected; rerun this same one-time request.
