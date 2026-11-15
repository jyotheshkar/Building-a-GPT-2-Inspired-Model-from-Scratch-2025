"""Bounded execution checks for the installed and frozen learning application."""
import platform
from pathlib import Path
import sys
import tempfile
from threading import Event

import torch

from . import __version__
from .checkpoint import load_checkpoint, save_checkpoint
from .config import ModelConfig, TrainingConfig
from .tokenizer import Tokenizer
from .training import generate, load_demo_corpus, train


def run_smoke():
    """Exercise real optimizer updates, checkpoint recovery, generation and Tk.

    This verifies execution only. No trained-policy, language-quality or broad
    desktop-compatibility result is implied by these bounded synthetic checks.
    """
    corpus = load_demo_corpus()
    vocabulary = Tokenizer.from_text(corpus)
    architecture = ModelConfig(vocab_size=vocabulary.vocab_size, block_size=16,
                               n_embd=16, n_head=2, n_layer=1, dropout=0.0)
    configuration = TrainingConfig(steps=3, batch_size=2, eval_every=3,
                                   eval_batches=1, seed=42, threads=1)
    stopped = Event()
    stopped.set()
    before = train(corpus, architecture, configuration, cancel_event=stopped)
    result = train(corpus, architecture, configuration)
    changed = any(not torch.equal(left, right) for left, right in
                  zip(before.model.parameters(), result.model.parameters()))
    finite = all(bool(torch.isfinite(parameter).all()) for parameter in result.model.parameters())
    with tempfile.TemporaryDirectory(prefix="gpt-scratch-smoke-") as temporary:
        checkpoint = save_checkpoint(Path(temporary)/"model.pt", result)
        restored = load_checkpoint(checkpoint)
        checkpoint_roundtrip = all(torch.equal(value, restored.model.state_dict()[key])
                                   for key, value in result.model.state_dict().items())
        tokenizer_roundtrip = restored.tokenizer.chars == result.tokenizer.chars
        checkpoint_roundtrip = checkpoint_roundtrip and restored.config == result.config
        first = generate(result, max_new_tokens=16, seed=11)
        second = generate(restored, max_new_tokens=16, seed=11)
        generation_finite = (first == second and len(second) == 17
                             and all(char in restored.tokenizer.chars for char in second))

    result.model.eval()
    indices = torch.arange(16).unsqueeze(0) % vocabulary.vocab_size
    changed_indices = indices.clone()
    changed_indices[:, -1] = (changed_indices[:, -1] + 1) % vocabulary.vocab_size
    with torch.inference_mode():
        original, _ = result.model(indices)
        altered, _ = result.model(changed_indices)
    causal = bool(torch.allclose(original[:, :-1], altered[:, :-1], atol=1e-6, rtol=0))
    from .gui import smoke_gui
    gui = smoke_gui()
    passed = all((changed, finite, checkpoint_roundtrip, tokenizer_roundtrip,
                  generation_finite, causal, result.steps_completed >= 2,
                  result.model.parameter_count == architecture.estimated_parameters,
                  gui.get("status") == "passed"))
    return {"status": "passed" if passed else "failed", "version": __version__,
            "frozen": bool(getattr(sys, "frozen", False)), "device": "cpu",
            "training_steps": result.steps_completed, "parameters_changed": changed,
            "parameters_finite": finite, "parameter_count": result.model.parameter_count,
            "checkpoint_roundtrip": checkpoint_roundtrip, "tokenizer_roundtrip": tokenizer_roundtrip,
            "generation_length": len(second), "generation_finite": generation_finite,
            "causal_mask_passed": causal, "gui": gui,
            "demo_corpus_sha256": result.corpus_sha256, "model_config": architecture.to_dict(),
            "python": platform.python_version(), "torch": str(torch.__version__)}
