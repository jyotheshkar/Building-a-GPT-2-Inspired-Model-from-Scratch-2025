"""Offline educational GPT-2-inspired character language-model lab."""

__version__ = "1.0.0"


def __getattr__(name):
    if name in {"ModelConfig", "TrainingConfig"}:
        from . import config
        return getattr(config, name)
    if name == "Tokenizer":
        from .tokenizer import Tokenizer
        return Tokenizer
    raise AttributeError(name)
