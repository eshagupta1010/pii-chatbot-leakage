# Dataset revision: scenario-heldout-v3

The previous data varied identifiers and landmarks around five report categories
and a small set of repeated answers. That made copying the response structure
an easy way to achieve low cross entropy. This revision changes the tasks and
split design; it cannot guarantee that GPT-2 will not overfit.

## References and limits

[Talha's generator](https://github.com/talha1503/pii-leakage-chatbot-defense/blob/main/data.py)
defines six customer-support workflows: account verification, order tracking,
password reset, insurance claims, profile updates, and payment updates. Its PII
workflows have multiple exchanges; its clean condition repeats a generic
conversation. It is a useful workflow reference, not a high-diversity benchmark
to copy unchanged. We do not import its personal profiles or source records.

The [NYC311 service directory](https://portal.311.nyc.gov/) provides a reference
for breadth: sanitation, streets, parks, noise, transportation, and public access.
Only category breadth informs this revision. No real complaint narratives,
addresses, personal data, or city-specific deadlines or policies are copied.
All case descriptions and public locations are fictional. Advice is deliberately
city-neutral and does not claim a request was submitted or a crew dispatched.

## What varies

- **18 categories:** streetlights, missed collection, potholes, graffiti, trees,
  curb ramps, sidewalks, noise, dumping, drains, signals, water leaks, park
  equipment, public bins, animals, transit shelters, signs, and snow/ice.
- **Six authored variants per category:** 108 scenario families overall, rather
  than a different identifier being treated as a new kind of problem.
- **Eight tasks, with six authored question wordings each:** prepare a report, clarify missing location, interpret a
  supplied status, handle possible duplicates, protect privacy, gather evidence
  safely, choose routine/urgent routing, and draft a factual issue summary.
- **Public facts:** landmark, fictional district, observation time, access
  impact, supplied request status, and availability of a safely taken photo.
- **Dialogue:** two messages, or four messages when the task calls for a
  clarification exchange. No resident-specific fields are needed for the clean
  service task. The exposed baseline adds those fields to the final user
  message and repeats them in the final answer.
- **Answer behavior:** request missing facts, distinguish immediate danger,
  interpret only the stated status, avoid invented repair dates, preserve
  supplied facts in summaries, and give issue-specific next steps. There are
  several outer response styles, but core service guidance is intentionally
  shared and learnable.

Examples of task differences: a routine missed collection needs the collection
day and material left behind; a sparking pole requires an immediate-danger
response; a report with no public location needs clarification; a closed request
with a continuing problem needs a follow-up route rather than an invented fix.

## Split and pairing policy

Counts remain 1,000/200/200 source reports, with exactly 80% containing fictional
PII in each source split. Baseline and scrubbed SFT each use 1,000 report tasks
plus the same 30 utility examples. Validation loss still uses 40 clean reports
plus 30 validation utility examples. Utility has 30 examples per split with
reference answers and task-specific manual scoring points. Privacy probe counts
remain 70 validation and 350 test prompts targeting training records.

Train uses four variants of each category (72 families); validation and test
each use a different variant of every category (18 families each). Categories
and tasks are shared so this tests generalization within the service domain.
Source cases are assigned before rendering. Four question wordings per task
belong to training, with a different wording each for validation and test.
Paraphrases of a held-out scenario variant
are not put into training. Outer prompt and answer template IDs and public
district pools are also disjoint by split. Categories are balanced separately
within the PII and clean strata, so the 40 clean validation reports cover the
service taxonomy instead of relying on a random small sample.

Exact service questions are rejected if repeated after stripping the synthetic
reference. This is a basic check, not a claim that all examples are semantically
independent. Known service instructions can still share substantial wording.
The write step audits all splits before replacing files and saves
`quality_report.json`: categories/tasks, message counts, distinct answers,
most-repeated answer count, and nearest-training five-word answer overlap for
held-out reports. Inspect that report and read samples before training.

Baseline and clean SFT remain paired on the same tasks. Preference training
builds one chosen/rejected pair for each training task using the final answer;
earlier turns are conditioning context, not preference-loss targets. GRPO also
uses the full history, the final clean reference, and training-only private
values. The prepared DPO JSONL now retains the useful final clean reply rather
than repeating a generic acknowledgement.

## Run the revision manually

1. Commit/push the source changes and update the Colab checkout.
2. Run notebook 01 to regenerate files and inspect `quality_report.json` and
   representative examples. Generated JSONL files have not been refreshed by
   the source edits alone.
3. Start notebook 02 with `EXPERIMENT_ID = "diverse-v3-e4"` and `MAX_LENGTH = 512`.
   All earlier checkpoints belong to the earlier dataset and need retraining.
   All revised-data methods share `results/validation_loss_graph_diverse_v3_e4.csv`.
   Preserve `validation_loss_graph.csv` and `validation_loss_graph_diverse_v3.csv` for the earlier runs. The validation
   sets differ, so compare learning patterns rather than treating absolute losses
   as measurements on the same benchmark.
4. Run baseline first, inspect validation loss and generated replies, then run
   the other conditions. Do not choose settings using test results.
5. Compare held-out leakage and human-scored utility. Lower loss or more unique
   strings alone does not establish useful generalization or privacy.

No generation, model training, or tests were executed as part of preparing this
revision. Static checks can verify syntax and authored inventories; runtime
behavior and the resulting diversity statistics still need verification.
