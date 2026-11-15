"""An explicit character vocabulary stored alongside every exported model."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Tokenizer:
    chars: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.chars, tuple):
            object.__setattr__(self, "chars", tuple(self.chars))
        if not 2 <= len(self.chars) <= 4096:
            raise ValueError("A vocabulary must contain 2 to 4,096 distinct characters.")
        if any(not isinstance(char, str) or len(char) != 1 for char in self.chars):
            raise ValueError("Each vocabulary entry must be exactly one character.")
        if len(set(self.chars)) != len(self.chars):
            raise ValueError("The vocabulary contains duplicate characters.")
        try:
            "".join(self.chars).encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("Vocabulary characters must be valid UTF-8 text.") from exc

    @classmethod
    def from_text(cls, text):
        if not isinstance(text, str):
            raise ValueError("The corpus must be text.")
        return cls(chars=tuple(sorted(set(text))))

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        if not isinstance(text, str):
            raise ValueError("The prompt must be text.")
        lookup = {char: number for number, char in enumerate(self.chars)}
        unknown = sorted(set(text) - lookup.keys())
        if unknown:
            names = ", ".join(repr(char) for char in unknown[:8])
            raise ValueError(f"Text contains characters absent from this model's vocabulary: {names}. Use characters from its training corpus.")
        return [lookup[char] for char in text]

    def decode(self, ids):
        values = list(ids)
        if any(type(value) is not int or not 0 <= value < self.vocab_size for value in values):
            raise ValueError("Token IDs must be integers inside this vocabulary.")
        return "".join(self.chars[value] for value in values)
