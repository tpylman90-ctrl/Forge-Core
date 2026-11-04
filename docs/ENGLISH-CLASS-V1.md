# Forge Seed — English Class v1: letters, words and sentences

## Why this experiment

The long-term Forge project is training a language model from **random
initialization**, not adopting another company's pretrained weights.
Previous TinyStories training established limited public-text learning,
but grammar and meaning assessments remained unreliable. Instead of
treating the Transformer as if it should learn school-level grammar
from raw text alone, we can provide an explicit original,
**structured English curriculum**.

This is an experiment, not a declaration that a small language model
now understands English. Natural reading also requires rich context,
conversation and generalization; forced-choice school exercises are
only one transparent diagnostic.

## Lesson ladder

The curriculum adapts the *existing* source-verified **Forge Seed
v0.7** language model, preserving the original scratch-trained
tokenizer and weights as an immutable source.

| Level | What it teaches | Measured exercise |
| --- | --- | --- |
| 1 | Alphabet and letter case | Match lowercase letters to capitals |
| 2 | Vowels and consonants | Identify the correct letter category |
| 3 | Building words | Join ordered written word parts |
| 4 | Nouns and verbs | Identify the action word in a short sentence |
| 5 | Subjects and objects | Identify who performs an action between two people |
| 6 | Complete sentences | Distinguish standard subject-verb-object order from a rearrangement |

Each source prompt begins with a short English-class rule and an
independently constructed example. The development and exam prompts
**do not repeat the lesson explanation**, and use distinct question
wordings. For later vocabulary-dependent tasks, training,
development and examination have disjoint sets of names, objects,
verbs and synthetic word parts. Alphabet categories necessarily
reuse the same English letters.

Every A/B correct answer is reconstructed from the source string
using explicit independently coded orthographic and grammatical
rules; a tampered answer or altered source fails closed.
Every category has equal A and B positions. This guards against
the common failure of choosing one letter for every question.

## Fair training comparisons

The experiment initializes three identical branches from the
**same authenticated original scratch-language checkpoint**:

- **English class:** six levels in order. Each level practices its
  present skill and retains some rehearsal from earlier levels.
- **Mixed lessons:** the exact same training batches and optimizer
  budget as English class, only shuffled across the schedule. This
  isolates lesson ordering while holding exposure constant.
- **Corrupted teaching:** the staged schedule with within-skill
  shuffled A/B labels (answer counts preserved). This estimates
  whether the verified answers convey signal beyond spurious
  preferences for a label.

For all three arms, **20% of each optimizer update** is rehearsal
on the original authenticated public TinyStories training text
and 80% on the supervised exercise. Public-language minibatches,
optimizer settings, initial weights and per-arm updates are matched.

The default pilot uses **120 optimizer steps at each of six levels**,
so **720 updates per arm**. Training is sequential on one CPU
GitHub Actions runner; job timeout is 50 minutes for setup, model
downloads, training, evaluation and artifacts. This is a bounded
one-off workload, not a claim of unlimited daily Actions CPU.

## Metrics, privacy and promotion gate

Before adaptation, evaluate the frozen model on development and
final original synthetic exam tasks, plus a held-out upstream
TinyStories validation loss. During training, choose each research
checkpoint using **development data only**, including the weakest
lesson category. Score the previously unused final synthetic exam
**after checkpoint selection**. Report per-category accuracy and A/B
prediction counts, including label-shuffle and mixed-order controls.

Measure retention as held-out public-story next-token loss versus
the frozen source. Quiz accuracy alone must **not** override a large
loss regression, and no two-choice score establishes conversation,
phonics pronunciation or independent language comprehension.

All experimental adapted model states are written to temporary
research artifacts. No canonical Forge Seed weights or tokenizer are
overwritten, and no experimental checkpoint is promoted to Android.
Nothing from the user's phone, chat history, application usage or
personal encrypted memories enters the public repository.

## Decision criteria

An encouraging result requires gains across multiple skills on
unseen wording and vocabulary, an advantage over shuffled labels,
and acceptable preservation of TinyStories language loss.
Improvement over mixed lesson ordering is needed before claiming
the **order** itself was useful.

For a later confirmation we should evaluate completely new grammar
constructions, narrative sentence generation, human-written examples
and at least three independent random initializations. The current
exercise family remains deterministic and synthetic, so even a strong
result is preliminary.

Next versions could include phoneme/grapheme patterns, plurals,
agreement, tense, negation, prepositions, paragraph coherence,
questions, and why-explanations. Actual sound-to-letter
correspondence would need carefully sourced pronunciation/audio
data; a text-only letter quiz does not teach listening or speaking.

Source: `src/forge_core/neural/english_class_v1.py`.
Tests: `tests/test_english_class_v1.py`.
Workflow: `.github/workflows/english-class-v1.yml`.
