# 311 Chatbot Privacy and Helpfulness

This project tests whether a chatbot can help with ordinary 311 questions without
revealing private details from fictional resident reports. It adapts the basic
experiment structure of [PII Leakage Analysis in Chatbot Settings](https://github.com/talha1503/pii-leakage-chatbot-defense):
prepare data, compare a baseline with privacy defenses, simulate extraction
attempts, and measure privacy alongside usefulness.

**All records and contact details generated here are fictional. Do not add real
resident data.** This is a small research prototype, not an evaluation of a real
city system or a claim about actual residents.

## Research question

When a language model has seen 311 reports containing personal details, can it
be prompted to repeat those details? Can a privacy defense reduce disclosure
while preserving helpful answers to general 311 questions?

## Day 1: data and test design

Requires Python 3.9 or newer. The starter dataset is generated locally using
only Python's standard library:

```bash
python src/generate_data.py
```

Or run [`notebooks/01_data_and_test_design.ipynb`](notebooks/01_data_and_test_design.ipynb)
from the project root (or open it in Jupyter/VS Code). No model download, GPU,
or external dataset is needed for Day 1.

The generator uses a fixed seed and creates:

| File | Contents |
|---|---|
| `data/synthetic/training_records.jsonl` | 1,000 fictional reports intended to be visible during baseline training. |
| `data/synthetic/validation_records.jsonl` | 200 disjoint fictional reports for validation and development checks. |
| `data/synthetic/test_records.jsonl` | 200 disjoint fictional reports reserved for final held-out checks. |
| `data/synthetic/training_conversations.jsonl` | 1,000 example filing conversations in which the assistant repeats report details; these provide a simple memorization-risk baseline. |
| `data/synthetic/scrubbed_training_conversations.jsonl` | Matched training conversations with names, contact details, and addresses replaced by `[REDACTED]`. |
| `data/synthetic/validation_conversations.jsonl` and `scrubbed_validation_conversations.jsonl` | Matched validation conversations for computing condition-appropriate validation loss. |
| `data/synthetic/test_conversations.jsonl` and `scrubbed_test_conversations.jsonl` | Matched held-out conversations reserved for later evaluation. |
| `data/synthetic/privacy_probes.jsonl` | 2,000 direct and indirect attempts to elicit details from training records, with expected answers for scoring. |
| `data/synthetic/utility_questions.jsonl` | General 311 questions that do not require resident-specific information. |

Each report has a generated name selected from combinations of a diverse
fictional first-name and last-name list, a unique number in the reserved
fictional-use `555-01xx` range, a name-based `example.com` email with a unique
suffix, and an invented address in Exampleton. These values are deliberately
synthetic and marked private for the experiment; real cities may collect,
publish, or protect different fields. The names are not taken from Talha's
project; this generator follows the same general synthetic-data approach
without copying anyone's records.

The split sizes now match the reference project's configured 1,000 training,
200 validation, and 200 test examples. This refers to our generated examples,
not a claim that the reference project's full dataset was run or validated.

## Planned experiment

1. **Baseline:** train or adapt a small language model on the synthetic filing
   conversations, including the private fields.
2. **Scrubbing defense:** train the same model on the matched conversations
   after replacing private values with `[REDACTED]`. This small-project
   adaptation preserves the service-task examples, unlike dropping every
   conversation that contains PII. It is a simple comparison, not a guarantee
   of privacy.
3. **Optional preference/reward defenses:** attempt DPO or GRPO only if the
   hardware, time, and libraries make them feasible. These are not required for
   the core result.
4. **Evaluation:** use the same fixed privacy probes and general 311 questions
   for each model. Report disclosure rate and utility separately, with sample
   counts and examples of errors.

The privacy probes target details the baseline has seen, so they test
memorization/extraction. Held-out reports are kept separate for later
generalization checks. General service questions measure whether a defense
simply makes the chatbot refuse everything.

## Four-day work plan

- **Day 1:** generate and inspect 1,000 training, 200 validation, and 200 test
  records; freeze the split and probe design.
- **Day 2:** run the baseline and scrubbing comparison, choosing the smallest
  setup that works on a laptop or free cloud notebook.
- **Day 3:** evaluate the same probes and utility questions for each model.
- **Day 4:** review failures, summarize results and limitations, and save
  reproducible commands/configuration.

## Day 2: model training setup

The training comparison uses the same GPT-2 base model for both conditions:
original synthetic conversations as the baseline, and matched conversations
with private values redacted as the defense. Resident messages remain model
context, but their tokens are masked from the training objective: GPT-2 is
optimized only to predict assistant replies. The redacted condition uses the
same prompts and replies with private values replaced by `[REDACTED]`.

Validation conversations are generated separately from the 200 validation
records. The baseline is validated on unredacted conversations and the
redacted model on the matched redacted conversations. Test records are never
used to fit or tune either model.

The script records an assistant-token-weighted train and validation loss after
each epoch. Each row in `results/training_metrics.csv` includes a unique `run_id`
shared by the baseline and redacted conditions, and records the sample count,
epoch, seed, device, and hyperparameters. Repeated runs append rows; do not
delete or overwrite prior results. The CSV is intentionally tracked by Git so
the results can be versioned. Checkpoints and run configs remain under ignored
`models/`.

For manual execution from the project root:

```bash
python -m pip install -r requirements.txt
python src/generate_data.py
python src/train_models.py --max-train-examples 250 --epochs 1 \
  --output-dir models/day2-answer-only/n250-e1
python src/train_models.py --epochs 1 \
  --output-dir models/day2-answer-only/nall-e1
python src/train_models.py --epochs 2 \
  --output-dir models/day2-answer-only/nall-e2
```

Each invocation trains both conditions and appends one CSV row per condition per
epoch. A 250-example run uses a deterministic random subset shared by both
conditions. The validation split remains at 200 examples. `--help` lists other
options.

### Prior runs and interpretation

The earlier Day 2 runs (including the 1,000-example run and a later
100-example notebook run) trained on both user and assistant tokens. That
objective could teach the model to reproduce resident messages, so those
checkpoints and reported losses are **superseded and should not be used** for
the corrected experiment. The corrected script uses a new output directory to
avoid overwriting those checkpoints. Start fresh with it; the corrected run
metrics will be appended to the CSV.

The reference project describes four model conditions (baseline, SFT with
scrubbing, DPO, and GRPO), and its default configuration specifies 200 test
examples and 100 attack queries. Its attack runner applies the configured
count to direct, indirect, and contextual attacks and half that count to
jailbreaks—350 attack prompts per model at that setting. The README also gives
smaller quick-run settings (100 test examples and 20 attack queries). These are
**documented/configured sizes**, not proof of how many runs were actually
completed: the repository labels its headline performance table as expected
results and does not provide a complete run log. Our core comparison is two
training conditions; we will increase rigor through held-out privacy and
utility evaluation, not by implying we reproduced all four training methods.

Training loss is only a diagnostic; privacy and usefulness must be measured
separately. A lower training loss does not by itself mean a better or safer
chatbot.

## Planned measures

- **Disclosure rate:** fraction of privacy probes where the expected private
  value appears in the response. We will also inspect false positives and
  misses because a simple string match is imperfect.
- **Utility:** fraction of general 311 questions answered usefully, scored with
  a transparent rubric or fixed expected-answer checks.
- **Trade-off:** show privacy and utility side by side rather than hiding one in
  a single score.

These are planned measures, not results. Any reported result should include the
number of records/probes, model and settings, and whether it was run locally or
in Colab. The redaction-based defense is related to, but not identical to,
filtering out PII-containing training examples in the reference project. Small
synthetic tests do not establish how real 311 systems behave.

## Project layout

```text
README.md
src/
  generate_data.py
notebooks/
  01_data_and_test_design.ipynb
data/
  synthetic/              # deterministic, fictional generated examples
results/                  # later evaluation outputs
```

## Reproducibility and safety

- Seed: `42` by default; change it with `python src/generate_data.py --seed 7`.
- The generated files are deterministic for a given seed and generator version.
- The starter generator has no third-party dependencies and makes no network
  calls.
- Keep personal, confidential, or real complaint information out of this
  repository.
- Later model results should be treated as a small controlled demonstration,
  not evidence that a specific real-world system is safe or unsafe.
