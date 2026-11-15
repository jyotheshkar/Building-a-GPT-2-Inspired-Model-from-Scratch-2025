"""Input and resource-boundary checks that do not need the ML runtime."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gpt_scratch_lab.config import ModelConfig, TrainingConfig
from gpt_scratch_lab.tokenizer import Tokenizer


class ConfigAndTokenizerTests(unittest.TestCase):
    def test_model_limits_reject_unbounded_or_incompatible_dimensions(self):
        for changes in ({"vocab_size": 1}, {"block_size": 513}, {"n_embd": 64, "n_head": 3},
                        {"n_layer": 8, "n_embd": 384}, {"n_embd": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ModelConfig(**changes)

    def test_training_rejects_nonfinite_or_unbounded_user_settings(self):
        for changes in ({"steps": 0}, {"steps": 10001}, {"batch_size": 129},
                        {"learning_rate": float("nan")}, {"learning_rate": float("inf")},
                        {"learning_rate": 0}, {"train_fraction": 1}, {"threads": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                TrainingConfig(**changes)

    def test_unicode_vocabulary_roundtrip_keeps_exact_character_order(self):
        text = "a\nβ🙂a"
        tokenizer = Tokenizer.from_text(text)
        restored = Tokenizer(chars=tokenizer.chars)
        self.assertEqual(restored.decode(tokenizer.encode(text)), text)
        self.assertEqual(restored.chars, tuple(sorted(set(text))))

    def test_unknown_prompt_character_has_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "absent.*vocabulary"):
            Tokenizer.from_text("abc abc").encode("D")

    def test_invalid_vocabulary_and_ids_are_rejected(self):
        for chars in (("a", "a"), ("ab", "c"), ("a",), ("a", "\ud800")):
            with self.subTest(chars=repr(chars)), self.assertRaises(ValueError):
                Tokenizer(chars=chars)
        tokenizer = Tokenizer.from_text("abc")
        for ids in ([-1], [3], [True], [1.0]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                tokenizer.decode(ids)


if __name__ == "__main__":
    unittest.main()
