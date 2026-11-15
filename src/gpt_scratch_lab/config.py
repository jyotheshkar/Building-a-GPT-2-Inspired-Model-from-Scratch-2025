"""Validated, bounded configurations; importing this module needs no PyTorch."""
from dataclasses import asdict, dataclass
import math

MAX_PARAMETERS = 12_000_000
MAX_CORPUS_BYTES = 4 * 1024 * 1024
MAX_GENERATED_TOKENS = 1024


def integer(name, value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}.")


def number(name, value, minimum, maximum, *, minimum_open=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value > maximum
            or (value <= minimum if minimum_open else value < minimum)):
        comparison = "greater than" if minimum_open else "at least"
        raise ValueError(f"{name} must be finite, {comparison} {minimum}, and at most {maximum}.")


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    block_size: int = 64
    n_embd: int = 64
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1

    def __post_init__(self):
        integer("vocab_size", self.vocab_size, 2, 4096)
        integer("block_size", self.block_size, 8, 512)
        integer("n_embd", self.n_embd, 16, 384)
        integer("n_head", self.n_head, 1, 12)
        integer("n_layer", self.n_layer, 1, 8)
        number("dropout", self.dropout, 0, 0.5)
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must divide evenly by n_head.")
        if self.estimated_parameters > MAX_PARAMETERS:
            raise ValueError(f"This configuration exceeds the {MAX_PARAMETERS:,}-parameter desktop limit.")

    @property
    def estimated_parameters(self):
        width = self.n_embd
        return (self.vocab_size * width + self.block_size * width
                + self.n_layer * (12 * width * width + 13 * width) + 2 * width)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 100
    batch_size: int = 8
    learning_rate: float = 3e-4
    train_fraction: float = 0.8
    eval_every: int = 25
    eval_batches: int = 2
    seed: int = 42
    threads: int = 1

    def __post_init__(self):
        integer("steps", self.steps, 1, 10_000)
        integer("batch_size", self.batch_size, 1, 128)
        number("learning_rate", self.learning_rate, 0, 1, minimum_open=True)
        number("train_fraction", self.train_fraction, 0.5, 0.95)
        integer("eval_every", self.eval_every, 1, 10_000)
        integer("eval_batches", self.eval_batches, 1, 32)
        integer("seed", self.seed, 0, 2**31 - 1)
        integer("threads", self.threads, 1, 8)

    def to_dict(self):
        return asdict(self)
