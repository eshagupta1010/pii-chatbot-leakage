"""Focused checks for the study logger, preferences, and optimization objectives."""

import csv
import math
import tempfile
import unittest
from pathlib import Path

from src.train_study import (
    STUDY_COLUMNS, _append_row, _dpo_loss, _dpo_objective, _format_prompt, _grpo_loss, _preference_pairs,
    _private_index, _reward, _reward_components,
)

try:
    import torch
except ImportError:
    torch = None


class StudyDataTests(unittest.TestCase):
    def test_all_conditions_append_to_one_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation_loss_graph.csv"
            for condition in ("baseline", "sft_filtered", "dpo", "grpo"):
                _append_row(path, {"experiment_id": "study-1", "condition": condition, "epoch": 1})
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
                self.assertEqual(tuple(reader.fieldnames), STUDY_COLUMNS)
            self.assertEqual([row["condition"] for row in rows], ["baseline", "sft_filtered", "dpo", "grpo"])

    def test_preferences_preserve_safe_answer_and_use_exposed_prompt(self):
        exposed = [{"record_id": "report-0001", "messages": [
            {"role": "user", "content": "My phone is 212-555-0102. How can I report a pothole?"},
            {"role": "assistant", "content": "Use 311. Phone: 212-555-0102."},
        ]}]
        safe = [{"record_id": "report-0001", "messages": [
            {"role": "user", "content": "How can I report a pothole?"},
            {"role": "assistant", "content": "Use 311 with the location and road condition."},
        ]}]
        pair = _preference_pairs(exposed, safe)[0]
        self.assertIn("212-555-0102", pair["prompt"])
        self.assertEqual(pair["chosen"], safe[0]["messages"][1]["content"])
        self.assertIn("212-555-0102", pair["rejected"])
        with self.assertRaisesRegex(ValueError, "matching lengths"):
            _preference_pairs(exposed, [])

    def test_preferences_keep_full_context_and_target_final_reply(self):
        context = [
            {"role": "user", "content": "Where should I report this?"},
            {"role": "assistant", "content": "What is the public location?"},
            {"role": "user", "content": "Near the public library. Phone: 212-555-0102."},
        ]
        exposed = [{"record_id": "report-0001", "messages": context + [
            {"role": "assistant", "content": "Use 311. Phone: 212-555-0102."}]}]
        safe = [{"record_id": "report-0001", "messages": context + [
            {"role": "assistant", "content": "Describe the issue near the public library to 311."}]}]
        pair = _preference_pairs(exposed, safe)[0]
        self.assertEqual(pair["context_messages"], context)
        self.assertIn("Assistant: What is the public location?", pair["prompt"])
        self.assertTrue(pair["prompt"].endswith("\nAssistant:"))
        self.assertEqual(pair["chosen"], safe[0]["messages"][-1]["content"])
        formatted = _format_prompt(context, "<stop>")
        self.assertIn("Assistant: What is the public location?<stop>\nUser:", formatted)

    def test_reward_prefers_helpful_safe_answer_to_leak_or_refusal(self):
        answer = "Use 311 and give the intersection and road damage."
        values = _private_index([("phone", "212-555-0102"), ("resident_name", "Maya Patel")])
        helpful = _reward(answer, answer, values)
        self.assertEqual(helpful, 1.0)
        self.assertLess(_reward(answer + " Call 212 555 0102.", answer, values), 0)
        self.assertEqual(_reward("I cannot help with city services.", answer, values), 0)
        self.assertEqual(_reward("", answer, values), -1)

    def test_reward_components_explain_disclosures_and_empty_answers(self):
        values = _private_index([("phone", "212-555-0102"), ("resident_name", "Maya Patel")])
        parts = _reward_components("Maya Patel: 212 555 0102", "Report the location to 311.", values)
        self.assertEqual(parts["leaked_values"], 2)
        self.assertEqual(parts["privacy_penalty"], 4)
        self.assertEqual(parts["reward"], parts["utility_reward"] - parts["privacy_penalty"])
        empty = _reward_components("", "Report the location to 311.", values)
        self.assertEqual(empty["empty_response"], 1)
        self.assertEqual(empty["reward"], -1)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ObjectiveTests(unittest.TestCase):
    def test_dpo_anchor_adds_a_gradient_for_the_chosen_answer(self):
        scores = torch.tensor([-6.0, -4.0], requires_grad=True)
        reference = scores.detach().clone()
        counts = torch.tensor([3, 2])
        pure, _ = _dpo_objective(scores, reference, counts, 0.1, 0.0, torch)
        anchored, metrics = _dpo_objective(scores, reference, counts, 0.1, 0.1, torch)
        pure_grad, = torch.autograd.grad(pure, scores, retain_graph=True)
        anchored_grad, = torch.autograd.grad(anchored, scores)
        self.assertAlmostEqual(anchored.item() - pure.item(), 0.2, places=6)
        self.assertAlmostEqual(anchored_grad[0].item() - pure_grad[0].item(), -0.1 / 3, places=6)
        self.assertAlmostEqual(anchored_grad[1].item(), pure_grad[1].item(), places=6)
        self.assertEqual(metrics["dpo_chosen_nll"].item(), 2)

    def test_dpo_starts_at_log_two_and_improves_with_chosen_margin(self):
        zero = torch.tensor([0.0])
        start = _dpo_loss(zero, zero, zero, zero, 0.1, torch)
        better = _dpo_loss(torch.tensor([2.0]), zero, zero, zero, 0.1, torch)
        self.assertAlmostEqual(start.item(), math.log(2), places=6)
        self.assertLess(better.item(), start.item())

    def test_grpo_positive_advantage_is_clipped(self):
        old = torch.tensor([[-2.0]])
        current = torch.tensor([[-2.0 + math.log(2)]], requires_grad=True)
        loss = _grpo_loss(current, old, current.detach(), torch.tensor([[True]]),
                          torch.tensor(1.0), 0.04, 0.2, torch)
        self.assertAlmostEqual(loss.item(), -1.2, places=6)
        loss.backward()
        self.assertAlmostEqual(current.grad.item(), 0.0, places=6)

    def test_grpo_masks_prompt_tokens(self):
        current = torch.tensor([[-9.0, -1.0]], requires_grad=True)
        reference = torch.tensor([[-2.0, -1.0]])
        loss = _grpo_loss(current, current.detach(), reference,
                          torch.tensor([[False, True]]), torch.tensor(0.0), 0.04, 0.2, torch)
        self.assertEqual(loss.item(), 0)


if __name__ == "__main__":
    unittest.main()
