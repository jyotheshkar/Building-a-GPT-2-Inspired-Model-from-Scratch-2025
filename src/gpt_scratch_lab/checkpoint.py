"""Versioned local checkpoints for models created by GPT Scratch Lab.

Load only files you created or trust. Restricted tensor loading and size checks
reduce risk; they are not a universal sandbox for hostile checkpoint files.
Optimizer resumption and legacy whole-model pickle files are not supported.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import math
import os
from pathlib import Path
import re
import tempfile
from typing import TYPE_CHECKING
import zipfile

import torch

from .config import ModelConfig, TrainingConfig
from .model import GPTLanguageModel
from .tokenizer import Tokenizer

if TYPE_CHECKING:
    from .training import TrainingResult

FORMAT_VERSION = 1
MAX_CHECKPOINT_BYTES = 160 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 4096
MAX_HISTORY_ITEMS = 10001
PAYLOAD_FIELDS = frozenset({
    "format_version", "model_config", "training_config", "vocabulary",
    "steps_completed", "history", "cancelled", "corpus_sha256", "state_dict",
})


def _require_supported_torch() -> None:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:\+[-.\w]+)?", str(torch.__version__))
    if not match or tuple(map(int, match.groups())) < (2, 15, 0):
        raise ValueError("Checkpoint operations require PyTorch 2.15.0 or newer; use this release's pinned environment.")


def _bounded_archive(path: Path) -> None:
    if not path.is_file():
        raise ValueError("Choose an existing checkpoint file, rather than a folder.")
    size = path.stat().st_size
    if not 0 < size <= MAX_CHECKPOINT_BYTES:
        raise ValueError("Checkpoint file must be nonempty and at most 160 MiB.")
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ARCHIVE_MEMBERS:
                raise ValueError("Checkpoint archive contains too many records.")
            if sum(entry.file_size for entry in entries) > MAX_CHECKPOINT_BYTES:
                raise ValueError("Checkpoint expanded data exceeds the 160 MiB limit.")
            if any(entry.flag_bits & 1 for entry in entries):
                raise ValueError("Encrypted checkpoint archives are not supported.")
    except zipfile.BadZipFile as exc:
        raise ValueError("Use a GPT Scratch Lab checkpoint; legacy whole-model pickle files are not supported.") from exc


def _history(value: object, steps_completed: int) -> list[dict]:
    if type(value) is not list or len(value) > MAX_HISTORY_ITEMS:
        raise ValueError("Checkpoint metric history is invalid or too large.")
    copied = []
    for entry in value:
        if type(entry) is not dict or len(entry) > 16:
            raise ValueError("Checkpoint metric history must contain small plain dictionaries.")
        item = {}
        for key, number in entry.items():
            if type(key) is not str or len(key) > 64:
                raise ValueError("Checkpoint metric field names are invalid.")
            if number is not None:
                if (type(number) not in (int, float) or abs(number) > 1e15
                        or not math.isfinite(number)):
                    raise ValueError("Checkpoint metric values must be finite numbers or null.")
            item[key] = number
        step = item.get("step")
        if "step" in item and (type(step) is not int or not 0 <= step <= steps_completed):
            raise ValueError("Checkpoint metric step is outside its recorded training run.")
        if "train_loss" in item:
            loss = item["train_loss"]
            if type(loss) not in (int, float) or loss < 0:
                raise ValueError("Checkpoint training loss must be a finite nonnegative number.")
        copied.append(item)
    return copied


def _validated_result(payload: object) -> TrainingResult:
    from .training import TrainingResult

    if type(payload) is not dict or set(payload) != PAYLOAD_FIELDS:
        raise ValueError("This is not a supported GPT Scratch Lab checkpoint payload.")
    if type(payload["format_version"]) is not int or payload["format_version"] != FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint format version.")
    if type(payload["model_config"]) is not dict or type(payload["training_config"]) is not dict:
        raise ValueError("Checkpoint configuration must contain plain dictionaries.")
    try:
        config = ModelConfig(**payload["model_config"])
        training_config = TrainingConfig(**payload["training_config"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint configuration is invalid or exceeds the model resource limits.") from exc
    vocabulary = payload["vocabulary"]
    if type(vocabulary) is not list or len(vocabulary) > 4096:
        raise ValueError("Checkpoint character vocabulary is invalid or too large.")
    try:
        tokenizer = Tokenizer(chars=tuple(vocabulary))
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint character vocabulary must contain distinct single characters.") from exc
    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError("Checkpoint vocabulary size does not match its model configuration.")
    steps = payload["steps_completed"]
    if type(steps) is not int or not 0 <= steps <= training_config.steps:
        raise ValueError("Checkpoint training step count is invalid.")
    if type(payload["cancelled"]) is not bool:
        raise ValueError("Checkpoint cancellation marker must be boolean.")
    corpus_hash = payload["corpus_sha256"]
    if type(corpus_hash) is not str or not re.fullmatch(r"[0-9a-f]{64}", corpus_hash):
        raise ValueError("Checkpoint corpus fingerprint is invalid.")
    history = _history(payload["history"], steps)
    state = payload["state_dict"]
    if not isinstance(state, Mapping) or any(type(key) is not str for key in state):
        raise ValueError("Checkpoint state must map model tensor names to tensors.")
    # Construct only a bounded CPU model. Preserve the caller's global RNG state.
    with torch.random.fork_rng(devices=[]):
        model = GPTLanguageModel(config)
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("Checkpoint tensor names do not match the configured model.")
    for key, reference in expected.items():
        tensor = state[key]
        if (type(tensor) is not torch.Tensor or tensor.device.type != "cpu"
                or tensor.layout != torch.strided or tensor.dtype != reference.dtype
                or tuple(tensor.shape) != tuple(reference.shape)):
            raise ValueError(f"Checkpoint tensor shape or type is invalid: {key}.")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"Checkpoint contains non-finite model values: {key}.")
    if not torch.equal(state["token_embedding.weight"], state["lm_head.weight"]):
        raise ValueError("Checkpoint tied token/output embedding tensors disagree.")
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError("Checkpoint weights do not match the configured model.") from exc
    model.eval()
    return TrainingResult(model=model, tokenizer=tokenizer, config=config,
                          training_config=training_config, steps_completed=steps,
                          history=history, cancelled=payload["cancelled"],
                          corpus_sha256=corpus_hash)


def save_checkpoint(path: str | Path, result: TrainingResult) -> Path:
    """Atomically save CPU tensors plus the exact tokenizer/configuration metadata."""
    _require_supported_torch()
    destination = Path(path).expanduser()
    if destination.is_dir():
        raise ValueError("Choose a checkpoint filename, rather than a folder.")
    payload = {
        "format_version": FORMAT_VERSION,
        "model_config": asdict(result.config),
        "training_config": asdict(result.training_config),
        "vocabulary": list(result.tokenizer.chars),
        "steps_completed": result.steps_completed,
        "history": result.history,
        "cancelled": result.cancelled,
        "corpus_sha256": result.corpus_sha256,
        "state_dict": {key: value.detach().cpu().clone() for key, value in result.model.state_dict().items()},
    }
    _validated_result(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + destination.name + ".", suffix=".tmp", dir=destination.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        _bounded_archive(temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def load_checkpoint(path: str | Path) -> TrainingResult:
    """Load an owned/trusted versioned checkpoint with restricted tensor loading.

    There is deliberately no unsafe loading fallback and no automatic allowlist.
    """
    _require_supported_torch()
    source = Path(path).expanduser()
    _bounded_archive(source)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("The checkpoint could not be loaded with restricted tensor loading. Open only files you created or trust.") from exc
    return _validated_result(payload)
