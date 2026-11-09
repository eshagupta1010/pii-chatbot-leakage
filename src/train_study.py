"""Run one condition of the four-method study; append shared loss diagnostics.

Notebook 02 is the user-facing entry point. No training happens on import.
DPO: https://arxiv.org/abs/2305.18290
GRPO: https://arxiv.org/abs/2402.03300 (outcome supervision)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__:
    from .evaluate_models import _normalize_value
    from .generate_data import DATASET_VERSION, PRIVATE_FIELDS
    from .train_models import (
        _encode_conversation, _make_collate_fn, _read_jsonl,
        _select_device, _select_training_conversations, _validate_training_data,
    )
else:
    from evaluate_models import _normalize_value
    from generate_data import DATASET_VERSION, PRIVATE_FIELDS
    from train_models import (
        _encode_conversation, _make_collate_fn, _read_jsonl,
        _select_device, _select_training_conversations, _validate_training_data,
    )


CONDITIONS = ("baseline", "sft_filtered", "dpo", "grpo")
STUDY_COLUMNS = (
    "experiment_id", "run_id", "condition", "epoch",
    "train_loss_assistant_tokens", "validation_loss_assistant_tokens",
    "objective_loss", "mean_reward", "zero_variance_group_fraction",
    "training_examples", "training_report_examples", "training_pii_report_examples",
    "diagnostic_training_examples", "validation_examples", "optimizer_steps_completed",
    "model_name", "initial_checkpoint", "learning_rate", "beta", "seed", "device",
    "run_started_at_utc", "dataset_fingerprint",
    "dpo_preference_loss", "dpo_chosen_nll", "dpo_rejected_nll",
    "dpo_preference_accuracy", "dpo_reward_margin", "dpo_sft_weight",
    "mean_utility_reward", "mean_privacy_penalty", "mean_leaked_values",
    "empty_response_fraction",
)


def _check_csv(path: Path) -> None:
    if path.exists() and path.stat().st_size:
        with path.open(newline="", encoding="utf-8") as handle:
            if tuple(next(csv.reader(handle), [])) != STUDY_COLUMNS:
                raise ValueError(f"Incompatible study CSV header: {path}")


def _append_row(path: Path, row: dict[str, Any]) -> None:
    _check_csv(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists() or not path.stat().st_size
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STUDY_COLUMNS, lineterminator="\n")
        if header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in STUDY_COLUMNS})


def _preference_pairs(
    baseline: list[dict[str, Any]], safe: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Use training-only paired tasks, including utility and non-PII reports."""
    if len(baseline) != len(safe):
        raise ValueError("Preference datasets must have matching lengths")
    pairs = []
    for exposed, scrubbed in zip(baseline, safe):
        if exposed["record_id"] != scrubbed["record_id"]:
            raise ValueError("Preference examples must have matching record IDs")
        context = exposed["messages"][:-1]
        chosen = scrubbed["messages"][-1]["content"]
        rejected = exposed["messages"][-1]["content"]
        if chosen == rejected:
            rejected = "I cannot help with any city service requests."
        pairs.append({
            "record_id": exposed["record_id"],
            "prompt": _format_prompt(context), "context_messages": context,
            "chosen": chosen, "rejected": rejected,
        })
    return pairs


def _format_prompt(messages: list[dict], eos_token: str = "") -> str:
    return "\n".join(
        f"{message['role'].capitalize()}: {message['content']}" +
        (eos_token if message["role"] == "assistant" else "")
        for message in messages
    ) + "\nAssistant:"


def _encode_answer(prompt: str, answer: str, tokenizer: Any, limit: int,
                   context_messages: list[dict] | None = None) -> dict:
    context = context_messages if context_messages is not None else [
        {"role": "user", "content": prompt},
    ]
    return _encode_conversation({"record_id": "preference-answer", "messages":
        context + [{"role": "assistant", "content": answer}]}, tokenizer, limit,
        last_assistant_only=True)


def _token_logps(model: Any, batch: dict, torch: Any) -> tuple[Any, Any]:
    """Next-token log probabilities, with only completion tokens unmasked."""
    labels = batch["labels"][:, 1:]
    mask = labels != -100
    logits = model(input_ids=batch["input_ids"],
                   attention_mask=batch["attention_mask"], use_cache=False).logits[:, :-1]
    # Avoid retaining a full vocabulary-sized log-softmax tensor.
    nll = torch.nn.functional.cross_entropy(
        logits.transpose(1, 2), labels, ignore_index=-100, reduction="none"
    )
    return -nll, mask


def _dpo_loss(chosen: Any, rejected: Any, ref_chosen: Any,
              ref_rejected: Any, beta: float, torch: Any) -> Any:
    margin = (chosen - rejected) - (ref_chosen - ref_rejected)
    return -torch.nn.functional.logsigmoid(beta * margin).mean()


def _dpo_objective(scores: Any, ref_scores: Any, token_counts: Any,
                   beta: float, sft_weight: float, torch: Any) -> tuple[Any, dict]:
    """DPO plus an explicit chosen-answer NLL term to preserve helpfulness.

    NLL diagnostics are token-normalized per answer, then averaged over pairs.
    Setting sft_weight=0 recovers the original DPO objective.
    """
    n = len(scores) // 2
    preference = _dpo_loss(scores[:n], scores[n:], ref_scores[:n], ref_scores[n:], beta, torch)
    nll = -scores / token_counts.clamp_min(1)
    margin = beta * ((scores[:n] - scores[n:]) - (ref_scores[:n] - ref_scores[n:]))
    metrics = {
        "dpo_preference_loss": preference,
        "dpo_chosen_nll": nll[:n].mean(),
        "dpo_rejected_nll": nll[n:].mean(),
        "dpo_preference_accuracy": (margin > 0).float().mean(),
        "dpo_reward_margin": margin.mean(),
    }
    return preference + sft_weight * metrics["dpo_chosen_nll"], metrics


def _grpo_loss(logps: Any, old_logps: Any, ref_logps: Any, mask: Any,
               advantage: Any, beta: float, clip: float, torch: Any) -> Any:
    ratio = (logps - old_logps).exp()
    surrogate = torch.minimum(
        ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage
    )
    delta = ref_logps - logps
    # Sampled nonnegative KL estimator used in outcome-supervised GRPO.
    kl = delta.exp() - delta - 1
    return ((-surrogate + beta * kl) * mask).sum() / mask.sum().clamp_min(1)


_STOP_WORDS = set(
    "a an the and or to of in on at for with your you it is its this that "
    "be as by can should through local city report give provide include".split()
)


def _content_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold())) - _STOP_WORDS


def _private_index(private_values: list[tuple[str, str]]) -> dict[str, tuple[str, ...]]:
    return {field: tuple({_normalize_value(value, field) for key, value in private_values
                          if key == field and value}) for field in PRIVATE_FIELDS}


def _reward_components(response: str, reference: str,
                       private_values: dict[str, tuple[str, ...]]) -> dict[str, float]:
    """Training proxy: reference-word F1 minus exact private-field penalties.

    Uses training answers only, never held-out utility rubrics or attack probes.
    This is a heuristic reward, not a reported human utility score or PII detector.
    """
    if not response.strip():
        return dict(reward=-1.0, utility_reward=0.0, privacy_penalty=0.0,
                    leaked_values=0.0, empty_response=1.0)
    answer_words, reference_words = _content_words(response), _content_words(reference)
    overlap = len(answer_words & reference_words)
    utility = 2 * overlap / max(1, len(answer_words) + len(reference_words))
    if any(phrase in response.casefold() for phrase in ("cannot help", "can't help", "unable to help")):
        utility = 0.0
    leaked = 0
    for field, values in private_values.items():
        normalized = _normalize_value(response, field)
        if field == "phone":
            leaked += sum(value in normalized for value in values)
        else:
            leaked += sum(f" {value} " in f" {normalized} " for value in values)
    penalty = 2.0 * leaked
    return dict(reward=utility - penalty, utility_reward=utility, privacy_penalty=penalty,
                leaked_values=float(leaked), empty_response=0.0)


def _reward(response: str, reference: str, private_values: dict[str, tuple[str, ...]]) -> float:
    return _reward_components(response, reference, private_values)["reward"]


def _write_config(path: Path, config: dict) -> None:
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def run_condition(args: argparse.Namespace) -> None:
    # Validate local data and dependencies before loading/downloading a model.
    baseline = _read_jsonl(args.data_dir / "training_conversations.jsonl")
    safe = _read_jsonl(args.data_dir / "sft_filtered_training_conversations.jsonl")
    validation = _read_jsonl(args.data_dir / "validation_conversations.jsonl")
    _validate_training_data(baseline, safe, validation)
    baseline = _select_training_conversations(baseline, args.max_train_examples, args.seed)
    safe = _select_training_conversations(safe, args.max_train_examples, args.seed)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(args.data_dir.glob("*.jsonl"))}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    shared = {
        "experiment_id": args.experiment_dir.name, "model_name": args.model_name,
        "dataset_version": DATASET_VERSION, "dataset_sha256": hashes,
        "seed": args.seed, "max_train_examples": args.max_train_examples,
        "max_length": args.max_length, "metrics_csv": str(args.metrics_csv.resolve()),
        "prompt_format": "User: <question>\nAssistant:",
    }
    manifest_path = args.experiment_dir / "experiment.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != shared:
            raise ValueError("Experiment settings/data changed; restore them or use a new experiment folder")
    checkpoint = args.experiment_dir / args.condition
    if checkpoint.exists() and any(checkpoint.iterdir()):
        raise ValueError(f"{checkpoint} already has output; choose a new experiment folder to rerun")
    _check_csv(args.metrics_csv)
    if args.metrics_csv.exists():
        with args.metrics_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if (row["experiment_id"] == args.experiment_dir.name and
                        row["condition"] == args.condition):
                    raise ValueError("This experiment/condition already has CSV rows; use a new experiment ID")

    initial = args.model_name
    parent_config = None
    if args.condition in {"dpo", "grpo"}:
        initial = str(args.experiment_dir / "baseline")
        parent_path = Path(initial) / "run_config.json"
        if not parent_path.is_file():
            raise ValueError("Run the baseline cell to completion before DPO or GRPO")
        parent_config = json.loads(parent_path.read_text())
        if (not parent_config.get("completed") or parent_config.get("dataset_sha256") != hashes
                or parent_config.get("run_id") != args.experiment_dir.name):
            raise ValueError("Baseline must be completed with this experiment's dataset")

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _select_device(torch)
    local = args.condition in {"dpo", "grpo"}
    tokenizer = AutoTokenizer.from_pretrained(initial, local_files_only=local)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(initial, local_files_only=local).to(device)
    context_limit = getattr(model.config, "max_position_embeddings", args.max_length)
    if args.max_length > context_limit:
        raise ValueError(f"max_length exceeds model context {context_limit}")
    collate = _make_collate_fn(torch, tokenizer.pad_token_id)

    def encoded(rows: list[dict]) -> list[dict]:
        return [_encode_conversation(row, tokenizer, args.max_length) for row in rows]

    train_rows = baseline if args.condition != "sft_filtered" else safe
    train_items, diagnostic_items, validation_items = encoded(train_rows), encoded(safe), encoded(validation)

    def loader(items: list, shuffle: bool = False) -> Any:
        return DataLoader(items, batch_size=args.batch_size, shuffle=shuffle,
                          generator=torch.Generator().manual_seed(args.seed), collate_fn=collate)

    def move(batch: dict) -> dict:
        return {key: value.to(device) for key, value in batch.items()}

    def ce_loss(items: list) -> float:
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in loader(items):
                logps, mask = _token_logps(model, move(batch), torch)
                total += float((-logps * mask).sum().item())
                count += int(mask.sum().item())
        if not count:
            raise ValueError("No assistant tokens in diagnostic data")
        return total / count

    # Freeze the starting baseline as the independent reference for each method.
    reference = None
    pair_items = []
    if local:
        reference = AutoModelForCausalLM.from_pretrained(initial, local_files_only=True).to(device)
        reference.requires_grad_(False)
        reference.eval()
    if args.condition == "dpo":
        for pair in _preference_pairs(baseline, safe):
            pair_items.append((
                _encode_answer(pair["prompt"], pair["chosen"], tokenizer, args.max_length, pair["context_messages"]),
                _encode_answer(pair["prompt"], pair["rejected"], tokenizer, args.max_length, pair["context_messages"]),
            ))

    # Check every GRPO prompt has room for its complete, untruncated rollout.
    prompts = [tokenizer.encode(_format_prompt(row["messages"][:-1], tokenizer.eos_token),
                                add_special_tokens=False) for row in baseline]
    if args.condition == "grpo" and any(
        len(prompt) + args.max_new_tokens > args.max_length for prompt in prompts
    ):
        raise ValueError("A GRPO prompt plus max_new_tokens exceeds max_length; increase max_length")
    records = _read_jsonl(args.data_dir / "training_records.jsonl")
    selected_ids = {row["record_id"] for row in baseline}
    private_values = _private_index([
        (field, row[field]) for row in records if row["record_id"] in selected_ids
        for field in PRIVATE_FIELDS if row.get(field)
    ])

    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        _write_config(manifest_path, shared)
    checkpoint.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    config = {
        **shared, "run_id": args.experiment_dir.name, "condition": args.condition,
        "initial_checkpoint": initial, "reference_checkpoint": initial if local else None,
        "run_started_at_utc": started, "epochs": args.epochs,
        "batch_size": args.batch_size, "gradient_accumulation_steps": args.accumulation_steps,
        "learning_rate": args.learning_rate, "beta": args.beta,
        "dpo_sft_weight": args.dpo_sft_weight if args.condition == "dpo" else 0.0,
        "training_objective": (
            "DPO preference loss plus weighted chosen-answer NLL" if args.condition == "dpo"
            else "GRPO clipped objective with reference KL" if args.condition == "grpo"
            else "supervised assistant-token cross entropy"
        ),
        "checkpoint_selection": "lowest common validation cross entropy among trained epochs; not a privacy ranking",
        "grpo_group_size": args.group_size, "grpo_updates": args.grpo_updates,
        "grpo_clip": args.clip, "max_new_tokens": args.max_new_tokens,
        "reward": "training-reference content-word F1 minus 2 per exact private-field disclosure",
        "loss_scope": "assistant tokens only; common clean training/validation CE diagnostics",
        "training_examples": len(train_rows), "validation_examples": len(validation),
        "training_record_ids": [r["record_id"] for r in train_rows if r["record_id"].startswith("report-")],
        "training_pii_record_ids": [r["record_id"] for r in baseline if r.get("contains_pii")]
            if args.condition != "sft_filtered" else [],
        "parent_baseline_epoch": parent_config["selected_checkpoint_epoch"] if parent_config else None,
        "completed": False,
    }
    _write_config(checkpoint / "run_config.json", config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    steps, best_loss, best_epoch = 0, float("inf"), None

    def log_epoch(epoch: int, objective: Any = "", reward: Any = "", zero: Any = "",
                  diagnostics: dict | None = None) -> None:
        nonlocal best_loss, best_epoch
        train_ce, val_ce = ce_loss(diagnostic_items), ce_loss(validation_items)
        if not all(math.isfinite(value) for value in (train_ce, val_ce)):
            raise ValueError("Non-finite diagnostic loss; stopping this run")
        # Epoch zero diagnoses initialization; select only trained epochs.
        if epoch and val_ce < best_loss:
            best_loss, best_epoch = val_ce, epoch
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
        _append_row(args.metrics_csv, {
            "experiment_id": args.experiment_dir.name, "run_id": args.experiment_dir.name,
            "condition": args.condition, "epoch": epoch,
            "train_loss_assistant_tokens": train_ce, "validation_loss_assistant_tokens": val_ce,
            "objective_loss": objective, "mean_reward": reward, "zero_variance_group_fraction": zero,
            "training_examples": len(train_rows),
            "training_report_examples": sum(r["record_id"].startswith("report-") for r in train_rows),
            "training_pii_report_examples": sum(bool(r.get("contains_pii")) for r in train_rows),
            "diagnostic_training_examples": len(safe), "validation_examples": len(validation),
            "optimizer_steps_completed": steps, "model_name": args.model_name,
            "initial_checkpoint": initial, "learning_rate": args.learning_rate,
            "beta": args.beta if local else "", "seed": args.seed, "device": str(device),
            "run_started_at_utc": started, "dataset_fingerprint": fingerprint,
            "dpo_sft_weight": args.dpo_sft_weight if args.condition == "dpo" else "",
            **(diagnostics or {}),
        })
        config.update(selected_checkpoint_epoch=best_epoch,
                      best_validation_loss_assistant_tokens=best_loss if best_epoch else None,
                      last_logged_epoch=epoch, last_validation_loss_assistant_tokens=val_ce,
                      optimizer_steps=steps)
        _write_config(checkpoint / "run_config.json", config)
        print(f"{args.condition}: epoch {epoch}/{args.epochs}; common train CE {train_ce:.4f}; "
              f"validation CE {val_ce:.4f}; selected epoch {best_epoch}", flush=True)

    def step() -> None:
        nonlocal steps
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        steps += 1

    print(f"Training only {args.condition} on {device}; CSV: {args.metrics_csv}", flush=True)
    log_epoch(0)
    rng = random.Random(args.seed)
    train_loader = loader(train_items, shuffle=True)
    for epoch in range(1, args.epochs + 1):
        objectives, rewards, zero_groups = [], [], []
        dpo_totals: dict[str, float] = {}
        dpo_pairs = 0
        reward_totals: dict[str, float] = {}
        optimizer.zero_grad(set_to_none=True)
        if args.condition in {"baseline", "sft_filtered"}:
            model.train()
            batches = iter(train_loader)
            for start in range(0, len(train_loader), args.accumulation_steps):
                window = [next(batches) for _ in range(min(args.accumulation_steps, len(train_loader) - start))]
                tokens = sum(int((b["labels"][:, 1:] != -100).sum()) for b in window)
                for batch in window:
                    logps, mask = _token_logps(model, move(batch), torch)
                    loss = (-logps * mask).sum() / tokens
                    loss.backward()
                    objectives.append((float((-logps * mask).sum().detach()), int(mask.sum())))
                step()
        elif args.condition == "dpo":
            model.eval()  # Disable dropout while retaining gradients for likelihood ratios.
            indices = list(range(len(pair_items)))
            rng.shuffle(indices)
            window_size = args.batch_size * args.accumulation_steps
            for start in range(0, len(indices), window_size):
                window = indices[start:start + window_size]
                for offset in range(0, len(window), args.batch_size):
                    chosen, rejected = zip(*(pair_items[i] for i in window[offset:offset + args.batch_size]))
                    combined = move(collate(list(chosen) + list(rejected)))
                    logps, mask = _token_logps(model, combined, torch)
                    scores = (logps * mask).sum(dim=1)
                    with torch.no_grad():
                        ref_logps, ref_mask = _token_logps(reference, combined, torch)
                        ref_scores = (ref_logps * ref_mask).sum(dim=1)
                    n = len(chosen)
                    loss, metrics = _dpo_objective(
                        scores, ref_scores, mask.sum(dim=1), args.beta, args.dpo_sft_weight, torch
                    )
                    (loss * n / len(window)).backward()
                    objectives.append((float(loss.detach()) * n, n))
                    for key, value in metrics.items():
                        dpo_totals[key] = dpo_totals.get(key, 0.0) + float(value.detach()) * n
                    dpo_pairs += n
                step()
        else:
            model.eval()  # Rollout and optimization use the same dropout-free policy.
            indices = list(range(len(prompts)))
            rng.shuffle(indices)
            for start in range(0, len(indices), args.accumulation_steps):
                rollout_groups = []
                for index in indices[start:start + args.accumulation_steps]:
                    prompt = torch.tensor([prompts[index]], dtype=torch.long, device=device)
                    with torch.no_grad():
                        generated = model.generate(
                            input_ids=prompt, attention_mask=torch.ones_like(prompt),
                            do_sample=True, temperature=1.0, top_k=0, top_p=1.0,
                            num_return_sequences=args.group_size, max_new_tokens=args.max_new_tokens,
                            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                        )
                    group, group_rewards = [], []
                    for sequence in generated:
                        completion = sequence[len(prompts[index]):].tolist()
                        if tokenizer.eos_token_id in completion:
                            completion = completion[:completion.index(tokenizer.eos_token_id) + 1]
                        ids = prompts[index] + completion
                        item = {"input_ids": ids, "labels": [-100] * len(prompts[index]) + completion}
                        response = tokenizer.decode(completion, skip_special_tokens=True)
                        components = _reward_components(response, safe[index]["messages"][-1]["content"], private_values)
                        group_rewards.append(components["reward"])
                        for key, value in components.items():
                            reward_totals[key] = reward_totals.get(key, 0.0) + value
                        batch = move(collate([item]))
                        with torch.no_grad():
                            old_logps, _ = _token_logps(model, batch, torch)
                            ref_logps, _ = _token_logps(reference, batch, torch)
                        group.append((item, old_logps.cpu(), ref_logps.cpu()))
                    values = torch.tensor(group_rewards, device=device)
                    std = values.std(unbiased=False)
                    advantages = (values - values.mean()) / (std + 1e-6)
                    zero_groups.append(float(std.item() < 1e-6))
                    rewards.extend(group_rewards)
                    rollout_groups.append((group, advantages))
                for _ in range(args.grpo_updates):
                    for group, advantages in rollout_groups:
                        for index, (item, old, ref) in enumerate(group):
                            logps, mask = _token_logps(model, move(collate([item])), torch)
                            loss = _grpo_loss(logps, old.to(device), ref.to(device), mask,
                                              advantages[index], args.beta, args.clip, torch)
                            (loss / (len(rollout_groups) * len(group))).backward()
                            objectives.append((float(loss.detach()), 1))
                    step()
                if start == 0 or (start // args.accumulation_steps + 1) % 10 == 0:
                    print(f"grpo: epoch {epoch}, sampled {min(start + args.accumulation_steps, len(indices))}"
                          f"/{len(indices)} training prompts", flush=True)
        objective = sum(value for value, _ in objectives) / max(1, sum(n for _, n in objectives))
        if not math.isfinite(objective):
            raise ValueError("Non-finite optimization objective")
        diagnostics = {key: value / dpo_pairs for key, value in dpo_totals.items()} if dpo_pairs else {}
        if rewards:
            diagnostics.update({
                "mean_utility_reward": reward_totals["utility_reward"] / len(rewards),
                "mean_privacy_penalty": reward_totals["privacy_penalty"] / len(rewards),
                "mean_leaked_values": reward_totals["leaked_values"] / len(rewards),
                "empty_response_fraction": reward_totals["empty_response"] / len(rewards),
            })
        log_epoch(epoch, objective,
                  sum(rewards) / len(rewards) if rewards else "",
                  sum(zero_groups) / len(zero_groups) if zero_groups else "", diagnostics)
        if diagnostics:
            print(f"{args.condition} diagnostics: {diagnostics}", flush=True)
    if local:
        # Preserve the final alignment policy for privacy/utility evaluation,
        # even if a previous epoch predicts clean reference answers better.
        final_dir = checkpoint / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        final_config = {**config, "completed": True, "selected_checkpoint_epoch": args.epochs,
                        "checkpoint_selection": "final trained alignment epoch",
                        "checkpoint_validation_loss_assistant_tokens": config["last_validation_loss_assistant_tokens"]}
        _write_config(final_dir / "run_config.json", final_config)
        config["final_checkpoint"] = "final"
    config["completed"] = True
    _write_config(checkpoint / "run_config.json", config)
    print(f"Saved {args.condition} epoch {best_epoch} to {checkpoint}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--metrics-csv", type=Path, default=Path("results/validation_loss_graph_diverse_v3_e4.csv"))
    parser.add_argument("--model-name", default="gpt2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--dpo-sft-weight", type=float, default=0.1,
                        help="Weight of chosen-answer NLL added to DPO; set 0 for pure DPO.")
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--grpo-updates", type=int, default=2)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()
    for name in ("epochs", "batch_size", "accumulation_steps", "max_length", "grpo_updates", "max_new_tokens"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.group_size < 2:
        parser.error("GRPO group size must be at least two")
    if args.max_train_examples is not None and args.max_train_examples < 1:
        parser.error("max train examples must be positive")
    if any(not math.isfinite(value) or value <= 0 for value in (args.learning_rate, args.beta)):
        parser.error("learning rate and beta must be finite and positive")
    if not 0 < args.clip < 1:
        parser.error("clip must be between zero and one")
    if not math.isfinite(args.dpo_sft_weight) or args.dpo_sft_weight < 0:
        parser.error("DPO SFT weight must be finite and nonnegative")
    run_condition(args)


if __name__ == "__main__":
    main()
