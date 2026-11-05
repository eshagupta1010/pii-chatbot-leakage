import csv
import tempfile
import unittest
from pathlib import Path

from src.generate_data import build_dataset
from src.train_models import METRIC_COLUMNS, _append_metrics, _encode_conversation


class FakeTokenizer:
    eos_token_id = 99

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]


class TrainingDataTests(unittest.TestCase):
    def test_loss_labels_include_assistant_text_not_user_text(self) -> None:
        conversation = {
            "record_id": "test-1",
            "messages": [
                {"role": "user", "content": "private user text"},
                {"role": "assistant", "content": "assistant answer"},
            ],
        }
        encoded = _encode_conversation(conversation, FakeTokenizer(), max_length=128)
        input_ids = encoded["input_ids"]
        labels = encoded["labels"]
        user_text = [ord(character) for character in "private user text"]
        assistant_answer = [ord(character) for character in "assistant answer"]

        self.assertEqual(len(input_ids), len(labels))
        user_start = input_ids.index(user_text[0])
        self.assertEqual(input_ids[user_start : user_start + len(user_text)], user_text)
        self.assertTrue(
            all(
                label == -100
                for label in labels[user_start : user_start + len(user_text)]
            )
        )
        self.assertIn(assistant_answer, self._contiguous_runs(labels))
        self.assertEqual(labels[-1], FakeTokenizer.eos_token_id)

    def test_pii_rate_and_filtered_sft_examples(self) -> None:
        dataset = build_dataset(
            seed=17,
            train_count=20,
            validation_count=10,
            test_count=10,
            validation_attack_counts={
                "direct": 2,
                "indirect": 2,
                "contextual": 2,
                "jailbreak": 2,
            },
            test_attack_counts={
                "direct": 4,
                "indirect": 4,
                "contextual": 4,
                "jailbreak": 4,
            },
        )
        training_ids = {row["record_id"] for row in dataset["training_records"]}
        validation_ids = {row["record_id"] for row in dataset["validation_records"]}
        self.assertTrue(training_ids.isdisjoint(validation_ids))
        self.assertEqual(
            sum(row["contains_pii"] for row in dataset["training_records"]), 16
        )
        self.assertEqual(
            sum(row["contains_pii"] for row in dataset["validation_records"]), 8
        )
        self.assertEqual(
            len(dataset["training_conversations"]), 20 + len(dataset["utility_train_qa"])
        )
        filtered = dataset["sft_filtered_training_conversations"]
        self.assertEqual(len(filtered), 4 + len(dataset["utility_train_qa"]))
        self.assertTrue(all(not row["contains_pii"] for row in filtered))
        train_pii_ids = {
            row["record_id"]
            for row in dataset["training_records"]
            if row["contains_pii"]
        }
        self.assertEqual(len(dataset["dpo_training_pairs"]), 16)
        self.assertTrue(
            all(pair["record_id"] in train_pii_ids for pair in dataset["dpo_training_pairs"])
        )

    def test_attack_probe_counts_and_known_training_targets(self) -> None:
        dataset = build_dataset(
            seed=31,
            train_count=40,
            validation_count=10,
            test_count=10,
            validation_attack_counts={
                "direct": 2,
                "indirect": 2,
                "contextual": 2,
                "jailbreak": 2,
            },
            test_attack_counts={
                "direct": 4,
                "indirect": 4,
                "contextual": 4,
                "jailbreak": 2,
            },
        )
        probes = dataset["test_privacy_probes"]
        self.assertEqual(len(probes), 14)
        self.assertEqual(
            {kind: sum(row["attack_type"] == kind for row in probes) for kind in (
                "direct", "indirect", "contextual", "jailbreak"
            )},
            {"direct": 4, "indirect": 4, "contextual": 4, "jailbreak": 2},
        )
        training_ids = {
            row["record_id"]
            for row in dataset["training_records"]
            if row["contains_pii"]
        }
        self.assertTrue(
            all(probe["target_record_id"] in training_ids for probe in probes)
        )

    def test_metrics_rows_append_and_write_header_once(self) -> None:
        row = {column: "" for column in METRIC_COLUMNS}
        row.update({"run_id": "run-1", "condition": "baseline", "epoch": 1})
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "nested" / "metrics.csv"
            _append_metrics(path, row)
            row["run_id"] = "run-2"
            _append_metrics(path, row)
            with path.open(newline="", encoding="utf-8") as metrics_file:
                reader = csv.DictReader(metrics_file)
                rows = list(reader)
            self.assertEqual([entry["run_id"] for entry in rows], ["run-1", "run-2"])
            self.assertEqual(reader.fieldnames, list(METRIC_COLUMNS))

    def test_metrics_reject_incompatible_existing_header(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "metrics.csv"
            path.write_text("run_id,old_metric\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incompatible header"):
                _append_metrics(path, {"run_id": "run-1"})

    @staticmethod
    def _contiguous_runs(values: list[int]) -> list[list[int]]:
        runs: list[list[int]] = []
        current_run: list[int] = []
        for value in values:
            if value == -100:
                if current_run:
                    runs.append(current_run)
                    current_run = []
            else:
                current_run.append(value)
        if current_run:
            runs.append(current_run)
        return runs


if __name__ == "__main__":
    unittest.main()
