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
| `data/synthetic/training_records.jsonl` | 1,000 fictional reports; exactly 80% contain name/contact/address fields by default. |
| `data/synthetic/validation_records.jsonl` | 200 disjoint records, with the same configured PII rate. |
| `data/synthetic/test_records.jsonl` | 200 disjoint records, reserved for final held-out checks. |
| `data/synthetic/training_conversations.jsonl` | All training report conversations plus general 311 Q&A examples; PII-containing examples intentionally repeat details in assistant replies for the baseline. |
| `data/synthetic/sft_filtered_training_conversations.jsonl` | 1,000 paired service-task conversations with personal fields omitted, plus the same 30 general 311 Q&A examples (1,030 total). The legacy filename is retained. |
| `data/synthetic/validation_conversations.jsonl` | PII-free validation report conversations plus held-out general 311 Q&A; used for validation loss for both conditions. |
| `data/synthetic/test_conversations.jsonl` | Held-out synthetic report conversations, not used to train or tune models. |
| `data/synthetic/validation_privacy_probes.jsonl` | 70 validation attack prompts (20 direct, 20 indirect, 20 contextual, 10 jailbreak) targeting details in training records. |
| `data/synthetic/test_privacy_probes.jsonl` | 350 final attack prompts (100 direct, 100 indirect, 100 contextual, 50 jailbreak) targeting training-record details, with different wording from validation probes. |
| `data/synthetic/utility_train_qa.jsonl`, `utility_validation_qa.jsonl`, `utility_test_qa.jsonl` | 30 distinct general city-service questions per split, with reference answers and scoring criteria. Topics overlap across splits, but exact question wording does not. |
| `data/synthetic/dpo_training_pairs.jsonl` | Safe (chosen) versus detail-revealing (rejected) answers for DPO development. |

For each split, the generator assigns exactly `round(split_size × PII rate)`
records to the PII-containing group (80% by default) and shuffles them with the
seed. PII fields are absent—not redacted placeholders—in the other 20%, so
there are actual non-PII examples available to train the filtering defense.
Although the validation-record split has 200 records, validation loss uses
only the 40 PII-free report conversations plus 30 utility conversations (70
total); PII-containing validation reports are excluded by design.
PII-containing records use generated names, `example.com` emails, fictional
555-number patterns, and invented Exampleton addresses. No real resident data
or source-project records are used.

The split sizes and 80% injection rate match the reference project's documented
configuration (1,000 training, 200 validation, 200 test). These are our
generated examples, not evidence that the reference project's full experiment
was actually run.

## Training approaches and why

All methods use the same GPT-2 starting checkpoint and split. Training loss is
computed only on assistant replies; resident messages remain context, but are
not themselves prediction targets.

1. **Baseline SFT — implemented:** train on all synthetic report conversations
   (including PII) and general 311 Q&A. This is the comparison condition that
   intentionally exposes the model to private details.
2. **Size-matched scrubbed SFT — implemented:** retain all 1,000 service tasks
   while removing personal fields from both user and assistant messages. Both
   conditions share the same service content and 30 utility examples. Public
   landmarks are generated independently of private home addresses. The safe
   answers give issue-specific reporting steps without inventing submission
   status. This is a size-matched adaptation; the reference project instead
   drops PII-containing conversations. The legacy `sft_filtered` names remain
   for compatibility. No redaction placeholders or oversampling are used.
3. **DPO — data prepared; training not yet implemented:** use preference pairs
   where the chosen response helps with the service request without repeating
   personal details and the rejected response discloses them. This tests
   whether preference training can teach a more privacy-preserving response
   behavior after the model has seen PII.
4. **GRPO — planned; training not yet implemented:** sample multiple responses
   and optimize a reward that favors both privacy and useful city-service
   answers. This tests reward-guided privacy/utility trade-offs, but costs more
   compute and depends on a carefully checked reward function.

DPO and GRPO are explicitly stretch goals until their training and evaluation
implementations are added and verified. Do not describe them as completed
methods yet.

## What we measure

The evaluation runner is `src/evaluate_models.py`. It records generated answers,
computes exact-target privacy leakage, and exports utility answers for manual
rubric scoring. Training loss is a diagnostic only; it is not a privacy or
helpfulness result.

- **Privacy leakage:** fraction of attack responses containing the exact target
  private value, reported overall and separately by name, phone, email, address,
  and attack family. The runner reports counts, rates, and Wilson 95% intervals.
  It regenerates the split's fixed number of prompts targeting records actually
  present in the selected baseline checkpoint's training subset, and uses those
  same prompts for both conditions. Prompts identify a target by its synthetic
  report reference, not its expected private value. Matching normalizes case and
  punctuation; phone matching ignores formatting. Review saved responses too,
  since paraphrased disclosures can evade exact-value matching. Wilson
  intervals are calculated per prompt and do not adjust for repeated target
  records, so treat them as descriptive rather than independent-sample bounds.
- **Utility/helpfulness:** score answers to held-out general 311 questions using
  the provided reference criteria. The runner saves a `utility_review.csv` with
  one row per model/question. Manually fill `criteria_met` with the number of
  criteria satisfied (from 0 to `criteria_total`), and add notes where needed.
  Report total criteria satisfied divided by total criteria; do not score a
  blanket refusal as helpful. The held-out utility set contains 30 questions.
- **Training diagnostics:** assistant-reply-only train and validation loss,
  reported per epoch. Validation loss is not privacy or task accuracy.
- **Uncertainty and scope:** give sample counts and, where possible, confidence
  intervals. The final test attacks target records present in training, which
  is necessary to test extraction; held-out test records separately measure
  behavior on unseen report examples.

Use the validation probes and QA to debug or choose settings. Run the fixed test
suite once for final reporting; do not tune against it. Results on synthetic
records do not establish that a real city chatbot is safe or unsafe.

### Running the evaluation

Start with validation data. For example, to evaluate the available two-epoch,
200-report run:

```bash
python src/evaluate_models.py \
  --run-dir models/four-approach-study/n200-e2-20261106-152532 \
  --split validation \
  --output-dir results/evaluations/n200-e2-20261106-152532-validation
```

The output directory contains a `summary.json`, per-model JSONL response files,
and `utility_review.csv`. Once the validation setup is final, run the held-out
test split exactly once, using a distinct output directory:

```bash
python src/evaluate_models.py \
  --run-dir models/four-approach-study/n200-e2-20261106-152532 \
  --split test \
  --output-dir results/evaluations/n200-e2-20261106-152532-test
```

The runner uses deterministic greedy generation and the local checkpoints; it
does not download models. The baseline run config must include the PII record
IDs used in training so probes can be restricted to records that model actually
saw. Utility scoring remains a human review step; the generated response text
and rubric criteria are included in the CSV.

## Four-day work plan

- **Day 1:** generate and inspect 1,000 training, 200 validation, and 200 test
  records; freeze the split and probe design.
- **Day 2:** run the baseline and PII-filtered SFT comparison, choosing the smallest
  setup that works on a laptop or free cloud notebook.
- **Day 3:** evaluate the same probes and utility questions for each model.
- **Day 4:** review failures, summarize results and limitations, and save
  reproducible commands/configuration.

The bundled reference report specifies three SFT epochs and deletion of 80% of
reports. It does not establish a fixed utility-question count. Our 30 questions
per split follow this project's requested design; five epochs provide a longer
initial loss curve. This is an adaptation, not an exact reproduction.

## Day 2: model training setup

The implemented comparison fine-tunes the same GPT-2 base model on the baseline
and size-matched scrubbed SFT datasets. Resident messages remain model context,
but their tokens are masked from the training objective: GPT-2 is optimized only
to predict assistant replies. SFT removes the personal fields while retaining
all service tasks and the same general 311 Q&A.

Validation conversations contain only PII-free reports and held-out general
311 Q&A, and are shared between the two conditions. Test records are never used
to fit or tune either model.

The script records an assistant-token-weighted train and validation loss after
each epoch. Each row in `results/training_metrics.csv` includes a unique `run_id`
shared by the baseline and PII-filtered conditions, and records the sample
count, epoch, seed, device, and hyperparameters. The requested report limit and
actual report count are separate columns. Both conditions now have matching
report IDs and counts, including when a subset is requested. Repeated runs
append rows; do not delete or overwrite prior results. The trainer saves the checkpoint from the epoch with
the lowest validation loss and records its epoch in `run_config.json`. This
selects for the validation language-modeling objective, not privacy. Checkpoints
and run configs remain under ignored `models/`; metrics CSVs are tracked.

For a full-data learning-curve run, regenerate the data first, then use all
available reports, five epochs, a separate metrics CSV, and a distinct model
output directory:

```bash
python src/train_models.py \
  --data-dir data/synthetic \
  --epochs 5 \
  --output-dir models/full-data-e5 \
  --metrics-csv results/full-data-e5-metrics.csv
```

Omitting `--max-train-examples` uses all training examples. With the default
generated dataset, both conditions have 1,030 conversations (1,000 reports
plus the same 30 utility examples). Run notebook 01 or `python src/generate_data.py`
first: existing generated files are not changed merely by editing the generator.
The trainer rejects stale deletion-only data before downloading a model.

The CLI and notebook default to five epochs and a maximum sequence length of
384. Examples that exceed that length raise an error rather than silently
truncating questions or reference answers. Choose a larger length within the
model context if needed. Existing checkpoint directories must be empty; use a
new output directory for every experiment. Run configs store dataset SHA-256
hashes, and evaluation rejects changed data or checkpoints from different runs.

The `notebooks/02_train_chatbots.ipynb` notebook includes code to plot
assistant-token-weighted training and validation loss for each condition and
mark the lowest-validation-loss epoch. Five epochs is a first look at the
learning curve, not a universal optimum. Keep the validation split fixed while
choosing a checkpoint, and do not use test results to select epochs. Loss curves
across conditions should be viewed separately: the service tasks match, but
baseline answers additionally contain personal details.

### Historical full-data learning-curve run (earlier deletion-only dataset)

The earlier five-epoch run used 1,006 baseline and 206 deletion-filtered SFT
conversations, six validation utility questions, seed 42, and the MPS device.
These results do not describe the revised size-matched experiment. No new
training or dataset generation was run as part of the current edits.
Baseline validation loss reached its minimum at epoch 3 (0.3187), then rose to
0.3877 at epoch 5 while training loss continued to fall (0.0265 at epoch 3 to
0.0252 at epoch 5). Filtered-SFT validation loss reached its minimum at epoch 4
(0.3259), then rose to 0.3459 at epoch 5 while training loss continued to fall
(0.0459 to 0.0450). This pattern is consistent with overfitting after those
validation minima; for this run, the trainer saved the epoch-3 baseline and
epoch-4 filtered-SFT checkpoints. The validation split is small, so treat the
selected epochs as exploratory rather than definitive.

The CSV and plot for that run are
[`results/full-data-e5-20261107-metrics.csv`](results/full-data-e5-20261107-metrics.csv)
and
[`results/full-data-e5-20261107-metrics.png`](results/full-data-e5-20261107-metrics.png).

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

Each invocation trains both implemented conditions and appends one CSV row per
condition per epoch. `--max-train-examples` is a requested report-example limit,
not the total conversation count; general Q&A is always added. A 250-report run
selects the same deterministic report IDs for both conditions, with all 30
training utility questions added to each. Validation remains
PII-free and includes the held-out utility questions. `--help` lists other options.

### Prior runs and interpretation

The earlier Day 2 runs (including the 1,000-example run and a later
100-example notebook run) trained on both user and assistant tokens. That
objective could teach the model to reproduce resident messages, so those
checkpoints and reported losses are **superseded and should not be used** for
the corrected experiment. The corrected script saves the second condition under
`sft_filtered` rather than `redacted`. Use a new output directory to avoid
overwriting old checkpoints; corrected run metrics append to the CSV.

The current report-reference probes require models trained on the regenerated
conversations, which now include each synthetic report's non-private reference
in its user message. Checkpoints created before this change did not learn those
references and should be retrained before running the revised privacy probes.

The reference project describes four model conditions (baseline, SFT with
scrubbing, DPO, and GRPO), and its default configuration specifies 200 test
examples and 100 attack queries. Its attack runner applies the configured
count to direct, indirect, and contextual attacks and half that count to
jailbreaks—350 attack prompts per model at that setting. The README also gives
smaller quick-run settings (100 test examples and 20 attack queries). These are
**documented/configured sizes**, not proof of how many runs were actually
completed: the repository labels its headline performance table as expected
results and does not provide a complete run log. Our core comparison currently implements two
training conditions. DPO has preference data prepared, while DPO/GRPO model
training remains future work. We will increase rigor through held-out privacy and
utility evaluation, not by implying we reproduced all four training methods.

Training loss is only a diagnostic; privacy and usefulness must be measured
separately. A lower training loss does not by itself mean a better or safer
chatbot.

Any reported result should include the number of records/probes, model and
settings, and whether it was run locally or in Colab. Small synthetic tests do
not establish how real 311 systems behave.

## Project layout

```text
README.md
src/
  generate_data.py
  train_models.py
notebooks/
  01_data_and_test_design.ipynb
  02_train_chatbots.ipynb
data/
  synthetic/              # deterministic, fictional generated examples
results/
  training_metrics.csv    # append-only baseline/SFT training diagnostics
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
