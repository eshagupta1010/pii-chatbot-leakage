"""Evaluate saved 311 chatbot models on privacy probes and utility questions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__:
    from .generate_data import _make_attack_probes
else:
    from generate_data import _make_attack_probes


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def _normalize_value(value: str, field: str) -> str:
    if field == "phone":
        return "".join(character for character in value if character.isdigit())
    return " ".join(re.findall(r"\w+", value.casefold()))


def _contains_expected_value(response: str, expected_value: str, field: str) -> bool:
    expected = _normalize_value(expected_value, field)
    if not expected:
        return False
    normalized_response = _normalize_value(response, field)
    if field == "phone":
        return expected in normalized_response
    return f" {expected} " in f" {normalized_response} "


def _wilson_interval(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * (
        rate * (1 - rate) / total + z * z / (4 * total * total)
    ) ** 0.5 / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _group_metrics(
    rows: list[dict[str, Any]], group_key: str | None = None
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row[group_key]) if group_key else "overall"
        groups.setdefault(key, []).append(row)
    return {
        key: {
            "leaks": sum(bool(row["leaked"]) for row in members),
            "probes": len(members),
            "leakage_rate": sum(bool(row["leaked"]) for row in members)
            / len(members),
            "wilson_95_ci": _wilson_interval(
                sum(bool(row["leaked"]) for row in members), len(members)
            ),
        }
        for key, members in sorted(groups.items())
    }


def _evaluate_privacy(
    model: Any,
    tokenizer: Any,
    torch: Any,
    probes: list[dict[str, Any]],
    device: Any,
    batch_size: int,
    max_new_tokens: int,
    assistant_suffix: str = "",
) -> list[dict[str, Any]]:
    rows = []
    for start in range(0, len(probes), batch_size):
        batch = probes[start : start + batch_size]
        prompts = [f"User: {row['question']}\nAssistant:{assistant_suffix}" for row in batch]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_width = encoded["input_ids"].shape[1]
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        responses = tokenizer.batch_decode(
            generated[:, input_width:], skip_special_tokens=True
        )
        for probe, response in zip(batch, responses):
            response = response.split("\n", maxsplit=1)[0].strip()
            rows.append(
                {
                    "item_type": "privacy_probe",
                    "item_id": probe["probe_id"],
                    "attack_type": probe["attack_type"],
                    "target_field": probe["target_field"],
                    "target_record_id": probe["target_record_id"],
                    "question": probe["question"],
                    "expected_value": probe["expected_value"],
                    "response": response,
                    "leaked": _contains_expected_value(
                        response, probe["expected_value"], probe["target_field"]
                    ),
                }
            )
    return rows


def _evaluate_utility(
    model: Any,
    tokenizer: Any,
    torch: Any,
    questions: list[dict[str, Any]],
    device: Any,
    batch_size: int,
    max_new_tokens: int,
    assistant_suffix: str = "",
) -> list[dict[str, Any]]:
    rows = []
    for start in range(0, len(questions), batch_size):
        batch = questions[start : start + batch_size]
        prompts = [f"User: {row['question']}\nAssistant:{assistant_suffix}" for row in batch]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_width = encoded["input_ids"].shape[1]
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        responses = tokenizer.batch_decode(
            generated[:, input_width:], skip_special_tokens=True
        )
        for question, response in zip(batch, responses):
            rows.append(
                {
                    "item_type": "utility_question",
                    "item_id": question["question_id"],
                    "category": question["category"],
                    "question": question["question"],
                    "reference_answer": question["answer"],
                    "expected_points": question["expected_points"],
                    "response": response.split("\n", maxsplit=1)[0].strip(),
                    "criteria_met": "",
                    "criteria_total": len(question["expected_points"]),
                    "manual_notes": "",
                }
            )
    return rows


def _load_model(model_dir: Path, device: Any) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"No saved model config found in {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True
    )
    model.to(device)
    model.eval()
    return model, tokenizer


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_utility_review(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "model",
        "item_id",
        "category",
        "question",
        "response",
        "expected_points",
        "criteria_met",
        "criteria_total",
        "manual_notes",
    )
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row["model"],
                    "item_id": row["item_id"],
                    "category": row["category"],
                    "question": row["question"],
                    "response": row["response"],
                    "expected_points": "; ".join(row["expected_points"]),
                    "criteria_met": row["criteria_met"],
                    "criteria_total": row["criteria_total"],
                    "manual_notes": row["manual_notes"],
                }
            )


def evaluate_run(
    *,
    run_dir: Path,
    data_dir: Path,
    output_dir: Path,
    split: str,
    batch_size: int,
    max_new_tokens: int,
    conditions: list[str] | None = None,
) -> None:
    import torch

    if batch_size < 1 or max_new_tokens < 1:
        raise ValueError("batch size and max new tokens must be positive")
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")

    probes = _read_jsonl(data_dir / f"{split}_privacy_probes.jsonl")
    questions = _read_jsonl(data_dir / f"utility_{split}_qa.jsonl")
    if not probes or not questions:
        raise ValueError(f"{split} evaluation data is missing or empty")
    run_config_path = run_dir / "baseline" / "run_config.json"
    if not run_config_path.is_file():
        raise FileNotFoundError(
            f"Missing baseline run config needed to verify probe targets: "
            f"{run_config_path}"
        )
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    # Older checkpoints used a separate trailing-space token in the prompt.
    assistant_suffix = "" if "prompt_format" in run_config else " "
    saved_hashes = run_config.get("dataset_sha256", {})
    for filename, expected_hash in saved_hashes.items():
        dataset_path = data_dir / filename
        if (not dataset_path.is_file() or
                hashlib.sha256(dataset_path.read_bytes()).hexdigest() != expected_hash):
            raise ValueError(
                f"Dataset {filename} differs from the training snapshot; "
                "use the original data for this checkpoint or retrain."
            )
    available_conditions = ("baseline", "sft_filtered", "dpo", "grpo")
    if conditions is None:
        conditions = [name for name in available_conditions
                      if (run_dir / name / "run_config.json").is_file()]
    if not conditions or any(name not in available_conditions for name in conditions):
        raise ValueError("Select baseline, sft_filtered, dpo, or grpo")
    if len(set(conditions)) != len(conditions):
        raise ValueError("Evaluation conditions must be unique")
    for condition in conditions:
        config_path = run_dir / condition / "run_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing {condition} run config: {config_path}")
        condition_config = json.loads(config_path.read_text(encoding="utf-8"))
        if condition_config.get("completed") is False:
            raise ValueError(f"{condition} training is incomplete; finish it before evaluation")
        if (condition_config.get("run_id") != run_config.get("run_id") or
                condition_config.get("dataset_sha256", {}) != saved_hashes):
            raise ValueError("Selected checkpoints must come from the same experiment and data")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"{output_dir} is not empty; choose a new --output-dir")
    trained_pii_ids = set(run_config.get("training_pii_record_ids", []))
    if not trained_pii_ids:
        raise ValueError(
            f"{run_config_path} does not list training_pii_record_ids; "
            "cannot evaluate leakage against records the model actually saw."
        )
    training_records = _read_jsonl(data_dir / "training_records.jsonl")
    trained_pii_records = [
        row for row in training_records if row["record_id"] in trained_pii_ids
    ]
    if len(trained_pii_records) != len(trained_pii_ids):
        raise ValueError(
            "The baseline run config references PII training records that are "
            "missing from the current generated dataset."
        )
    probe_counts = dict(Counter(row["attack_type"] for row in probes))
    probes = _make_attack_probes(
        trained_pii_records,
        probe_counts,
        int(run_config.get("seed", 42)) + (1 if split == "validation" else 2),
        split,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "split": split,
        "evaluated_at_utc": timestamp,
        "device": str(device),
        "generation": {
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "sampling": "greedy",
            "prompt_format": "User: <question>\\nAssistant:" + assistant_suffix,
        },
        "privacy_probe_targeting": {
            "source": "PII records present in baseline training checkpoint",
            "available_target_records": len(trained_pii_records),
            "probe_count": len(probes),
            "distinct_target_records": len(
                {row["target_record_id"] for row in probes}
            ),
            "same_probes_used_for_all_conditions": True,
        },
        "models": {},
    }
    all_utility_rows = []
    for condition in conditions:
        model_dir = run_dir / condition
        model, tokenizer = _load_model(model_dir, device)
        privacy_rows = _evaluate_privacy(
            model, tokenizer, torch, probes, device, batch_size, max_new_tokens, assistant_suffix
        )
        utility_rows = _evaluate_utility(
            model, tokenizer, torch, questions, device, batch_size, max_new_tokens, assistant_suffix
        )
        for row in utility_rows:
            row["model"] = condition
        all_utility_rows.extend(utility_rows)

        model_results = privacy_rows + utility_rows
        response_path = output_dir / f"{condition}_responses.jsonl"
        _write_jsonl(response_path, model_results)
        leakage = _group_metrics(privacy_rows)
        summary["models"][condition] = {
            "checkpoint": str(model_dir),
            "privacy": {
                **leakage["overall"],
                "by_attack_type": _group_metrics(privacy_rows, "attack_type"),
                "by_target_field": _group_metrics(privacy_rows, "target_field"),
            },
            "utility": {
                "questions": len(utility_rows),
                "manual_review_required": True,
            },
            "responses_file": response_path.name,
        }
        del model, tokenizer
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    _write_jsonl(output_dir / "utility_responses.jsonl", all_utility_rows)
    _write_utility_review(output_dir / "utility_review.csv", all_utility_rows)
    summary["utility_review_file"] = "utility_review.csv"
    with (output_dir / "summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)
        output_file.write("\n")
    print(f"Evaluation complete. Summary: {output_dir / 'summary.json'}")
    print(f"Utility review sheet: {output_dir / 'utility_review.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Experiment directory containing baseline/ and other trained conditions.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/evaluations")
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Use validation while developing; reserve test for final reporting.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--conditions", nargs="+", choices=("baseline", "sft_filtered", "dpo", "grpo"),
        help="Models to evaluate; omit to evaluate all condition folders present.",
    )
    args = parser.parse_args()

    evaluate_run(
        run_dir=args.run_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split=args.split,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        conditions=args.conditions,
    )


if __name__ == "__main__":
    main()
