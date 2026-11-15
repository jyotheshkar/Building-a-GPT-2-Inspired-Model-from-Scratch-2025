"""Real checkpoint round trips and rejection before unsafe or ambiguous loads."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from gpt_scratch_lab import checkpoint
    from gpt_scratch_lab.config import ModelConfig, TrainingConfig
    from gpt_scratch_lab.training import train, generate

CORPUS = ("A small synthetic test passage supplies a predictable character vocabulary.\n" * 30)


@unittest.skipUnless(HAS_TORCH, "Install the pinned PyTorch runtime to exercise checkpoint behavior")
class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = ModelConfig(vocab_size=len(set(CORPUS)), block_size=8,
                             n_embd=16, n_head=2, n_layer=1, dropout=0.0)
        training = TrainingConfig(steps=2, batch_size=2, eval_every=2, eval_batches=1, threads=1)
        cls.result = train(CORPUS, config, training)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "model.pt"
        checkpoint.save_checkpoint(self.path, self.result)

    def payload(self):
        return torch.load(self.path, map_location="cpu", weights_only=True)

    def test_roundtrip_retains_exact_weights_vocabulary_metadata_and_generation(self):
        restored = checkpoint.load_checkpoint(self.path)
        self.assertEqual(restored.tokenizer.chars, self.result.tokenizer.chars)
        self.assertEqual(restored.config, self.result.config)
        self.assertEqual(restored.training_config, self.result.training_config)
        self.assertEqual(restored.steps_completed, self.result.steps_completed)
        self.assertEqual(restored.history, self.result.history)
        self.assertEqual(restored.corpus_sha256, self.result.corpus_sha256)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(restored.model.parameters(), self.result.model.parameters())))
        self.assertEqual(generate(restored, "A", max_new_tokens=12, seed=10),
                         generate(self.result, "A", max_new_tokens=12, seed=10))

    def test_legacy_pickle_is_rejected_before_tensor_loader(self):
        self.path.write_bytes(b"unsupported legacy pickle fixture")
        with patch.object(checkpoint.torch, "load") as load:
            with self.assertRaisesRegex(ValueError, "legacy"):
                checkpoint.load_checkpoint(self.path)
            load.assert_not_called()

    def test_unknown_format_or_vocabulary_mismatch_is_rejected(self):
        original = self.payload()
        payload = dict(original, format_version=2)
        torch.save(payload, self.path)
        with self.assertRaisesRegex(ValueError, "format version"):
            checkpoint.load_checkpoint(self.path)
        payload = dict(original, vocabulary=list(original["vocabulary"]) + ["☃"])
        torch.save(payload, self.path)
        with self.assertRaisesRegex(ValueError, "vocabulary size"):
            checkpoint.load_checkpoint(self.path)

    def test_wrong_tensor_shape_or_dtype_is_rejected(self):
        original = self.payload()
        key = next(iter(original["state_dict"]))
        for value in (original["state_dict"][key].reshape(-1), original["state_dict"][key].double()):
            payload = dict(original, state_dict=dict(original["state_dict"]))
            payload["state_dict"][key] = value
            torch.save(payload, self.path)
            with self.assertRaisesRegex(ValueError, "shape or type"):
                checkpoint.load_checkpoint(self.path)

    def test_nonfinite_and_disagreeing_tied_weights_are_rejected(self):
        original = self.payload()
        for value, message in ((float("nan"), "non-finite"), (1.0, "tied")):
            payload = dict(original, state_dict=dict(original["state_dict"]))
            tensor = payload["state_dict"]["lm_head.weight"].clone()
            if value != value:
                tensor[0, 0] = value
            else:
                tensor[0, 0] += value
            payload["state_dict"]["lm_head.weight"] = tensor
            torch.save(payload, self.path)
            with self.assertRaisesRegex(ValueError, message):
                checkpoint.load_checkpoint(self.path)

    def test_loader_uses_cpu_weights_only_and_never_retries_unsafely(self):
        with patch.object(checkpoint.torch, "load", wraps=torch.load) as load:
            checkpoint.load_checkpoint(self.path)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(load.call_args.kwargs, {"map_location": "cpu", "weights_only": True})
        with patch.object(checkpoint.torch, "load", side_effect=RuntimeError("restricted load fixture")) as load:
            with self.assertRaisesRegex(ValueError, "restricted tensor loading"):
                checkpoint.load_checkpoint(self.path)
            self.assertEqual(load.call_count, 1)

    def test_oversized_checkpoint_is_rejected_before_loading(self):
        with patch.object(checkpoint, "MAX_CHECKPOINT_BYTES", 1), patch.object(checkpoint.torch, "load") as load:
            with self.assertRaisesRegex(ValueError, "160 MiB"):
                checkpoint.load_checkpoint(self.path)
            load.assert_not_called()

    def test_invalid_save_does_not_replace_the_previous_checkpoint(self):
        before = self.path.read_bytes()
        invalid = replace(self.result, corpus_sha256="not a fingerprint")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            checkpoint.save_checkpoint(self.path, invalid)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_null_known_history_fields_cannot_reach_the_gui(self):
        original = self.payload()
        for field, value in (("step", None), ("train_loss", None), ("train_loss", -1.0)):
            payload = dict(original, history=[dict(original["history"][0])])
            payload["history"][0][field] = value
            torch.save(payload, self.path)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                checkpoint.load_checkpoint(self.path)


if __name__ == "__main__":
    unittest.main()
