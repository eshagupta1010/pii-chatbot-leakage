"""Fine-tune baseline and size-matched PII-scrubbed GPT-2 models."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__:
    from .generate_data import DATASET_VERSION
else:
    from generate_data import DATASET_VERSION


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def _select_training_conversations(
    conversations: list[dict[str, Any]], max_report_examples: int | None, seed: int
) -> list[dict[str, Any]]:
    report_conversations = [
        row for row in conversations if row["record_id"].startswith("report-")
    ]
    other_conversations = [
        row for row in conversations if not row["record_id"].startswith("report-")
    ]
    if max_report_examples is None:
        selected_reports = report_conversations
    else:
        selected_count = min(max_report_examples, len(report_conversations))
        indices = sorted(
            random.Random(seed).sample(range(len(report_conversations)), selected_count)
        )
        selected_reports = [report_conversations[index] for index in indices]
    return selected_reports + other_conversations


def _encode_conversation(
    conversation: dict[str, Any], tokenizer: Any, max_length: int,
    *, last_assistant_only: bool = False,
) -> dict[str, list[int]]:
    input_ids: list[int] = []
    labels: list[int] = []
    if not conversation["messages"]:
        raise ValueError("A training conversation must contain messages")
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token")

    for index, message in enumerate(conversation["messages"]):
        role = message["role"].lower()
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported role {role!r} in {conversation.get('record_id')}")

        prefix_ids = tokenizer.encode(
            f"{role.capitalize()}:", add_special_tokens=False
        )
        content_ids = tokenizer.encode(
            f" {message['content']}", add_special_tokens=False
        )
        separator_ids = tokenizer.encode("\n", add_special_tokens=False)

        input_ids.extend(prefix_ids)
        labels.extend([-100] * len(prefix_ids))
        input_ids.extend(content_ids)
        is_target = role == "assistant" and (not last_assistant_only or index == len(conversation["messages"]) - 1)
        if is_target:
            labels.extend(content_ids)
        else:
            labels.extend([-100] * len(content_ids))
        # Every assistant turn needs a stopping target, including clarifications.
        if role == "assistant":
            input_ids.append(eos_token_id)
            labels.append(eos_token_id if is_target else -100)
        if index < len(conversation["messages"]) - 1:
            input_ids.extend(separator_ids)
            labels.extend([-100] * len(separator_ids))

    if len(input_ids) > max_length:
        raise ValueError(
            f"{conversation.get('record_id')} needs {len(input_ids)} tokens, "
            f"exceeding max_length={max_length}; increase --max-length rather "
            "than silently truncating the question or reference answer."
        )
    if not any(label != -100 for label in labels[1:]):
        raise ValueError(
            f"No assistant response tokens fit within max_length={max_length} "
            f"for {conversation.get('record_id')}"
        )
    return {"input_ids": input_ids, "labels": labels}


def _select_device(torch: Any) -> Any:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _make_collate_fn(torch: Any, pad_token_id: int) -> Any:
    def collate(examples: list[dict[str, list[int]]]) -> dict[str, Any]:
        max_length = max(len(example["input_ids"]) for example in examples)
        input_ids = []
        attention_masks = []
        labels = []

        for example in examples:
            ids = example["input_ids"]
            example_labels = example["labels"]
            padding_length = max_length - len(ids)
            input_ids.append(ids + [pad_token_id] * padding_length)
            attention_masks.append([1] * len(ids) + [0] * padding_length)
            labels.append(example_labels + [-100] * padding_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def _train_one_model(
    *,
    name: str,
    conversations: list[dict[str, Any]],
    model_name: str,
    output_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    accumulation_steps: int,
    max_length: int,
    learning_rate: float,
    validation_conversations: list[dict[str, Any]],
    run_id: str,
    run_started_at: str,
    metrics_path: Path,
    train_examples_requested: int | None,
    dataset_hashes: dict[str, str],
) -> None:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    class ConversationDataset(Dataset):
        def __init__(self, examples: list[dict[str, Any]], tokenizer: Any) -> None:
            self.items = []
            for example in examples:
                self.items.append(
                    _encode_conversation(example, tokenizer, max_length)
                )

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            return self.items[index]

    print(f"\nTraining {name} with {len(conversations)} conversations.")
    device = _select_device(torch)
    print(f"Device: {device}; base model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if context_limit is not None and max_length > context_limit:
        raise ValueError(f"max_length={max_length} exceeds model context {context_limit}")
    model.to(device)

    dataset = ConversationDataset(conversations, tokenizer)
    validation_dataset = ConversationDataset(validation_conversations, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=_make_collate_fn(torch, tokenizer.pad_token_id),
    )
    train_eval_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_make_collate_fn(torch, tokenizer.pad_token_id),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_make_collate_fn(torch, tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")
    best_epoch = 0

    def evaluate(data_loader: Any) -> float:
        model.eval()
        total_loss = 0.0
        total_target_tokens = 0
        with torch.no_grad():
            for evaluation_batch in data_loader:
                target_tokens = int(
                    (evaluation_batch["labels"][:, 1:] != -100).sum().item()
                )
                if target_tokens == 0:
                    continue
                evaluation_batch = {
                    key: value.to(device) for key, value in evaluation_batch.items()
                }
                loss = model(**evaluation_batch).loss
                total_loss += loss.item() * target_tokens
                total_target_tokens += target_tokens
        if total_target_tokens == 0:
            raise ValueError("Evaluation data contains no assistant response tokens")
        model.train()
        return total_loss / total_target_tokens

    model.train()
    optimizer_steps = 0
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        batch_iterator = iter(loader)
        for window_start in range(0, len(loader), accumulation_steps):
            window = [
                next(batch_iterator)
                for _ in range(min(accumulation_steps, len(loader) - window_start))
            ]
            window_target_tokens = sum(
                int((batch["labels"][:, 1:] != -100).sum().item())
                for batch in window
            )
            for batch in window:
                target_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
                batch = {key: value.to(device) for key, value in batch.items()}
                loss = model(**batch).loss
                (loss * target_tokens / window_target_tokens).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        training_loss = evaluate(train_eval_loader)
        validation_loss = evaluate(validation_loader)
        if not all(math.isfinite(value) for value in (training_loss, validation_loss)):
            raise ValueError(f"Non-finite loss for {name} at epoch {epoch + 1}")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
        print(
            f"{name}: epoch {epoch + 1}/{epochs}, answer-only "
            f"training loss {training_loss:.4f}, validation loss {validation_loss:.4f}; "
            f"best checkpoint epoch {best_epoch}"
        )
        _append_metrics(
            metrics_path,
            {
                "run_id": run_id,
                "run_started_at_utc": run_started_at,
                "condition": name,
                "model_name": model_name,
                "training_examples": len(conversations),
                "training_report_examples_requested": (
                    train_examples_requested
                    if train_examples_requested is not None
                    else "all"
                ),
                "training_report_examples": sum(
                    row["record_id"].startswith("report-") for row in conversations
                ),
                "training_pii_report_examples": sum(
                    bool(row.get("contains_pii"))
                    for row in conversations
                    if row["record_id"].startswith("report-")
                ),
                "validation_examples": len(validation_conversations),
                "validation_report_examples": sum(
                    row["record_id"].startswith("report-")
                    for row in validation_conversations
                ),
                "epoch": epoch + 1,
                "train_loss_assistant_tokens": f"{training_loss:.6f}",
                "validation_loss_assistant_tokens": f"{validation_loss:.6f}",
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "gradient_accumulation_steps": accumulation_steps,
                "max_length": max_length,
                "seed": seed,
                "device": str(device),
                "optimizer_steps_completed": optimizer_steps,
            },
        )

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as output_file:
        json.dump(
            {
                "run_id": run_id,
                "run_started_at_utc": run_started_at,
                "dataset_version": DATASET_VERSION,
                "dataset_sha256": dataset_hashes,
                "condition": name,
                "data_preparation": (
                    "paired service tasks with fictional personal details"
                    if name == "baseline"
                    else "paired service tasks with personal fields removed"
                ),
                "model_name": model_name,
                "seed": seed,
                "epochs": epochs,
                "selected_checkpoint_epoch": best_epoch,
                "best_validation_loss_assistant_tokens": best_validation_loss,
                "training_examples": len(conversations),
                "max_train_report_examples": train_examples_requested,
                "validation_examples": len(validation_conversations),
                "batch_size": batch_size,
                "gradient_accumulation_steps": accumulation_steps,
                "max_length": max_length,
                "learning_rate": learning_rate,
                "loss_scope": "assistant response tokens only",
                "prompt_format": "User: <question>\nAssistant:",
                "metrics_csv": str(metrics_path),
                "training_record_ids": [
                    row["record_id"]
                    for row in conversations
                    if row["record_id"].startswith("report-")
                ],
                "training_pii_record_ids": [
                    row["record_id"]
                    for row in conversations
                    if row["record_id"].startswith("report-")
                    and row.get("contains_pii")
                ],
                "device": str(device),
                "optimizer_steps": optimizer_steps,
            },
            output_file,
            indent=2,
        )
        output_file.write("\n")
    print(
        f"Saved {name} best-validation checkpoint (epoch {best_epoch}) "
        f"and run configuration to {output_dir}"
    )

    del model, tokenizer, optimizer, loader, train_eval_loader
    del validation_loader, dataset, validation_dataset
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


METRIC_COLUMNS = (
    "run_id",
    "run_started_at_utc",
    "condition",
    "model_name",
    "training_examples",
    "training_report_examples_requested",
    "training_report_examples",
    "training_pii_report_examples",
    "validation_examples",
    "validation_report_examples",
    "epoch",
    "train_loss_assistant_tokens",
    "validation_loss_assistant_tokens",
    "learning_rate",
    "batch_size",
    "gradient_accumulation_steps",
    "max_length",
    "seed",
    "device",
    "optimizer_steps_completed",
)


def _append_metrics(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        with path.open(encoding="utf-8", newline="") as existing_file:
            existing_columns = next(csv.reader(existing_file), [])
        if tuple(existing_columns) != METRIC_COLUMNS:
            raise ValueError(
                f"Metrics CSV at {path} has an incompatible header; "
                "start a new CSV or migrate the existing file before appending."
            )
    with path.open("a", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=METRIC_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _validate_training_data(
    baseline: list[dict[str, Any]],
    scrubbed: list[dict[str, Any]],
    validation: list[dict[str, Any]],
) -> None:
    """Reject stale deletion-only datasets before any model is downloaded."""
    if not baseline or not scrubbed or not validation:
        raise ValueError("Empty dataset; run notebook 01 or src/generate_data.py first")
    baseline_ids = [row["record_id"] for row in baseline]
    scrubbed_ids = [row["record_id"] for row in scrubbed]
    if len(set(baseline_ids)) != len(baseline_ids) or baseline_ids != scrubbed_ids:
        raise ValueError(
            "Baseline and scrubbed SFT must have matching unique report/utility IDs; "
            "regenerate data with notebook 01 or src/generate_data.py."
        )
    if any(row.get("contains_pii") for row in scrubbed + validation):
        raise ValueError("Scrubbed training and validation must contain no PII reports")
    reports = [row for row in scrubbed if row["record_id"].startswith("report-")]
    if any(row.get("dataset_version") != DATASET_VERSION for row in reports):
        raise ValueError("Outdated SFT data; regenerate it before training")
    validation_reports = [row for row in validation if row["record_id"].startswith("report-")]
    for key in ("scenario_family", "prompt_template_id", "question_template_id", "answer_template_id"):
        training_groups = {row.get(key) for row in reports}
        validation_groups = {row.get(key) for row in validation_reports}
        if None in training_groups or None in validation_groups or training_groups & validation_groups:
            raise ValueError(f"Missing or overlapping {key}; regenerate the held-out scenario dataset")
    train_questions = {
        row["messages"][0]["content"] for row in baseline
        if row["record_id"].startswith("utility-")
    }
    validation_questions = {
        row["messages"][0]["content"] for row in validation
        if row["record_id"].startswith("utility-")
    }
    if len(train_questions) != 30 or len(validation_questions) != 30:
        raise ValueError("Expected 30 unique utility questions per split; regenerate data")
    if train_questions & validation_questions or set(baseline_ids) & {
        row["record_id"] for row in validation
    }:
        raise ValueError("Training and validation data overlap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/day2-answer-only")
    )
    parser.add_argument(
        "--metrics-csv", type=Path, default=Path("results/training_metrics.csv")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--max-train-examples",
        type=int,
        help="Select a deterministic random training subset; omit to use all records.",
    )
    parser.add_argument(
        "--max-validation-examples",
        type=int,
        help="Limit validation records for a quick smoke test; omit for all records.",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error("epochs, batch size, and accumulation steps must be positive")
    if args.max_length < 2:
        parser.error("max length must be at least 2")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be finite and positive")
    if args.max_train_examples is not None and args.max_train_examples < 1:
        parser.error("max train examples must be at least 1")
    if args.max_validation_examples is not None and args.max_validation_examples < 1:
        parser.error("max validation examples must be at least 1")

    baseline = _read_jsonl(args.data_dir / "training_conversations.jsonl")
    filtered = _read_jsonl(
        args.data_dir / "sft_filtered_training_conversations.jsonl"
    )
    validation = _read_jsonl(args.data_dir / "validation_conversations.jsonl")
    _validate_training_data(baseline, filtered, validation)
    for condition in ("baseline", "sft_filtered"):
        checkpoint_dir = args.output_dir / condition
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            raise ValueError(f"{checkpoint_dir} is not empty; choose a new --output-dir")
    if args.metrics_csv.exists() and args.metrics_csv.stat().st_size:
        with args.metrics_csv.open(encoding="utf-8", newline="") as metrics_file:
            if tuple(next(csv.reader(metrics_file), [])) != METRIC_COLUMNS:
                raise ValueError("Metrics CSV has an incompatible header; choose a new CSV")
    dataset_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.data_dir.glob("*.jsonl"))
    }

    if args.max_train_examples is not None:
        baseline = _select_training_conversations(
            baseline, args.max_train_examples, args.seed
        )
        filtered = _select_training_conversations(
            filtered, args.max_train_examples, args.seed
        )
    if args.max_validation_examples is not None:
        selected_count = min(args.max_validation_examples, len(validation))
        selected_indices = sorted(
            random.Random(args.seed + 1).sample(
                range(len(validation)), selected_count
            )
        )
        validation = [validation[index] for index in selected_indices]

    run_id = uuid.uuid4().hex
    run_started_at = datetime.now(timezone.utc).isoformat()
    _train_one_model(
        name="baseline",
        conversations=baseline,
        model_name=args.model_name,
        output_dir=args.output_dir / "baseline",
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        validation_conversations=validation,
        run_id=run_id,
        run_started_at=run_started_at,
        metrics_path=args.metrics_csv,
        train_examples_requested=args.max_train_examples,
        dataset_hashes=dataset_hashes,
    )
    _train_one_model(
        name="sft_filtered",
        conversations=filtered,
        model_name=args.model_name,
        output_dir=args.output_dir / "sft_filtered",
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        validation_conversations=validation,
        run_id=run_id,
        run_started_at=run_started_at,
        metrics_path=args.metrics_csv,
        train_examples_requested=args.max_train_examples,
        dataset_hashes=dataset_hashes,
    )


if __name__ == "__main__":
    main()
