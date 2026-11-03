# Forge Seed v0.7 — compositional short-answer experiment 001

Authorized one-time, bounded CPU learning experiment to address poor
generalization in the earlier 256-example direct-word-answer pilot.

- SOURCE: audited original 14,000-step TinyStories/BPE Forge Seed v0.7
  (Actions run 41572257363), with 2,219,136 original neural parameters.
- Generate 3,200 deterministic, rule-verified training questions across
  5 skills: color, recipient, giver, location, and count.
- Staged exposure of 160, 800 and 3,200 distinct examples, with
  250, 650, and 1,600 optimization updates respectively per arm.
- Compare two independent models starting from identical original weights:
  correct direct-word answers versus within-skill shuffled answer words.
- Equal training prompts, batch schedules, optimizer settings, and public
  TinyStories language-loss rehearsal (5%, 15%, 25%) for both arms.
- New training and challenge **person names, objects, narrative templates,
  question templates** are disjoint. Keep a separate 50-question progress
  set and 200-question final challenge that is not scored until all
  training is complete; 40 final questions per skill.
- Preserve per-skill exact direct-answer accuracy, wrong output samples,
  fixed-seed story samples, and held-out language token loss.
- No candidate is promoted, released as an assistant, installed in Android,
  or mixed with private personal memory. No pretrained model weights,
  personal observations or model-generated self-approved labels.
- The challenge is synthetic and internally authored. Its findings
  require replication on independently sourced natural questions and
  multiple seeds before declaring preschool-like comprehension.
