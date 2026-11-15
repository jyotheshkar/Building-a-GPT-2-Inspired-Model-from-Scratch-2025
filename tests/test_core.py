"""Actual causal-model, optimizer, generation and cancellation contracts."""
import importlib.util
import math
from pathlib import Path
import sys
from threading import Event
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from gpt_scratch_lab.config import ModelConfig, TrainingConfig
    from gpt_scratch_lab.model import GPTLanguageModel
    from gpt_scratch_lab.training import train, generate

CORPUS = ("A test robot records a short observation.\nAnother row supplies character context.\n" * 30)


@unittest.skipUnless(HAS_TORCH, "Install the pinned PyTorch runtime to exercise model behavior")
class CoreModelTests(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(vocab_size=len(set(CORPUS)), block_size=8,
                                  n_embd=16, n_head=2, n_layer=1, dropout=0.0)
        self.training = TrainingConfig(steps=3, batch_size=2, learning_rate=0.001,
                                       eval_every=2, eval_batches=1, seed=17, threads=1)

    def test_future_tokens_cannot_change_earlier_logits(self):
        torch.manual_seed(9)
        model = GPTLanguageModel(self.config).eval()
        tokens = torch.arange(8).unsqueeze(0) % self.config.vocab_size
        changed = tokens.clone()
        changed[:, 5:] = (changed[:, 5:] + 1) % self.config.vocab_size
        with torch.inference_mode():
            original_logits, _ = model(tokens)
            changed_logits, _ = model(changed)
        torch.testing.assert_close(original_logits[:, :5], changed_logits[:, :5], rtol=0, atol=1e-6)
        self.assertFalse(torch.allclose(original_logits[:, 5:], changed_logits[:, 5:]))
        self.assertEqual(model.parameter_count, self.config.estimated_parameters)

    def test_real_training_updates_parameters_and_records_finite_loss(self):
        stopped = Event()
        stopped.set()
        before = train(CORPUS, self.config, self.training, cancel_event=stopped)
        after = train(CORPUS, self.config, self.training)
        self.assertEqual(before.steps_completed, 0)
        self.assertTrue(before.cancelled)
        self.assertEqual(after.steps_completed, 3)
        self.assertFalse(after.cancelled)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before.model.parameters(), after.model.parameters())))
        self.assertTrue(all(bool(torch.isfinite(p).all()) for p in after.model.parameters()))
        self.assertTrue(after.history)
        self.assertTrue(all(math.isfinite(row["train_loss"]) for row in after.history))

    def test_same_seed_and_configuration_reproduce_cpu_training(self):
        first = train(CORPUS, self.config, self.training)
        second = train(CORPUS, self.config, self.training)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first.model.parameters(), second.model.parameters())))

    def test_cancel_after_first_progress_returns_partial_model(self):
        cancellation = Event()
        seen = []
        def progress(record):
            seen.append(record["step"])
            cancellation.set()
        result = train(CORPUS, self.config, self.training, progress=progress, cancel_event=cancellation)
        self.assertEqual(seen, [1])
        self.assertEqual(result.steps_completed, 1)
        self.assertTrue(result.cancelled)

    def test_generation_is_bounded_and_restores_the_callers_model_mode(self):
        stopped = Event()
        stopped.set()
        result = train(CORPUS, self.config, self.training, cancel_event=stopped)
        result.model.train()
        torch.set_num_threads(2)
        first = generate(result, prompt="A", max_new_tokens=7, seed=8)
        self.assertEqual(torch.get_num_threads(), result.training_config.threads)
        self.assertTrue(result.model.training)
        self.assertEqual(len(first), 8)
        self.assertTrue(all(parameter.grad is None for parameter in result.model.parameters()))
        result.model.eval()
        second = generate(result, prompt="A", max_new_tokens=7, seed=8)
        self.assertFalse(result.model.training)
        self.assertEqual(first, second)

    def test_quadratic_attention_budget_rejects_large_batch_before_model_allocation(self):
        architecture = ModelConfig(vocab_size=2, block_size=512, n_embd=32,
                                   n_head=8, n_layer=8, dropout=0.0)
        settings = TrainingConfig(steps=1, batch_size=128)
        with patch("gpt_scratch_lab.training.GPTLanguageModel") as constructor:
            with self.assertRaisesRegex(ValueError, "attention/activation budget"):
                train("ab" * 2000, architecture, settings)
            constructor.assert_not_called()

    def test_short_corpus_and_unknown_prompt_fail_understandably(self):
        with self.assertRaises(ValueError):
            train("abc", training_config=self.training)
        result = train(CORPUS, self.config, self.training)
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            generate(result, prompt="☃")
        for setting in ({"temperature": 0}, {"temperature": float("nan")},
                        {"max_new_tokens": 1025}, {"top_k": 0}):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                generate(result, **setting)


if __name__ == "__main__":
    unittest.main()
