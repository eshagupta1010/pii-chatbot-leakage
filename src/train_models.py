"""Fine-tune matched GPT-2 models on original and redacted 311 conversations."""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def _conversation_text(conversation: dict[str, Any]) -> str:
    return "\n".join(
        f"{message['role'].capitalize()}: {message['content']}"
        for message in conversation["messages"]
    )


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
            padding_length = max_length - len(ids)
            input_ids.append(ids + [pad_token_id] * padding_length)
            attention_masks.append([1] * len(ids) + [0] * padding_length)
            labels.append(ids + [-100] * padding_length)

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
                encoded = tokenizer(
                    _conversation_text(example),
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=True,
                )
                if not encoded["input_ids"]:
                    raise ValueError(f"Conversation {example.get('record_id')} is empty")
                self.items.append({"input_ids": encoded["input_ids"]})

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
    model.to(device)

    dataset = ConversationDataset(conversations, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=_make_collate_fn(torch, tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    optimizer_steps = 0
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for batch_index, batch in enumerate(loader):
            batch_start = (batch_index // accumulation_steps) * accumulation_steps
            current_window_size = min(
                accumulation_steps, len(loader) - batch_start
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            total_loss += loss.detach().item()
            (loss / current_window_size).backward()

            end_of_window = (
                (batch_index + 1) % accumulation_steps == 0
                or batch_index + 1 == len(loader)
            )
            if end_of_window:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

        average_loss = total_loss / len(loader)
        print(
            f"{name}: epoch {epoch + 1}/{epochs}, "
            f"mean training loss {average_loss:.4f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as output_file:
        json.dump(
            {
                "model_name": model_name,
                "seed": seed,
                "epochs": epochs,
                "training_examples": len(conversations),
                "batch_size": batch_size,
                "gradient_accumulation_steps": accumulation_steps,
                "max_length": max_length,
                "learning_rate": learning_rate,
                "device": str(device),
                "optimizer_steps": optimizer_steps,
            },
            output_file,
            indent=2,
        )
        output_file.write("\n")
    print(f"Saved {name} model and run configuration to {output_dir}")

    del model, tokenizer, optimizer, loader, dataset
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--output-dir", type=Path, default=Path("models/day2"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--max-train-examples",
        type=int,
        help="Limit training records for a quick smoke test; omit for all records.",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error("epochs, batch size, and accumulation steps must be positive")
    if args.max_length < 2:
        parser.error("max length must be at least 2")
    if args.max_train_examples is not None and args.max_train_examples < 1:
        parser.error("max train examples must be at least 1")

    baseline = _read_jsonl(args.data_dir / "training_conversations.jsonl")
    scrubbed = _read_jsonl(args.data_dir / "scrubbed_training_conversations.jsonl")
    if len(baseline) != len(scrubbed):
        raise ValueError("Baseline and scrubbed data must contain the same examples")
    if not baseline:
        raise ValueError("Training data is empty; generate the Day 1 dataset first")

    if args.max_train_examples is not None:
        baseline = baseline[: args.max_train_examples]
        scrubbed = scrubbed[: args.max_train_examples]

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
    )
    _train_one_model(
        name="redacted",
        conversations=scrubbed,
        model_name=args.model_name,
        output_dir=args.output_dir / "redacted",
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
