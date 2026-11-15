"""Bounded offline CPU training and generation shared by CLI and desktop UI."""
from dataclasses import dataclass, replace
import hashlib
from importlib import resources
import time

import torch

from .config import MAX_CORPUS_BYTES, MAX_GENERATED_TOKENS, ModelConfig, TrainingConfig, integer, number
from .model import GPTLanguageModel
from .tokenizer import Tokenizer


@dataclass
class TrainingResult:
    model: GPTLanguageModel
    tokenizer: Tokenizer
    config: ModelConfig
    training_config: TrainingConfig
    steps_completed: int
    history: list[dict]
    cancelled: bool
    corpus_sha256: str


def load_demo_corpus():
    return resources.files("gpt_scratch_lab").joinpath("data/demo.txt").read_text(encoding="utf-8")


def _cancelled(event):
    return event is not None and event.is_set()


def _batch(data, batch_size, block_size, generator):
    if len(data) <= block_size:
        raise ValueError("Each data split needs more characters than the context length.")
    starts = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    inputs = torch.stack([data[start:start + block_size] for start in starts])
    targets = torch.stack([data[start + 1:start + block_size + 1] for start in starts])
    return inputs, targets


def _validation_loss(model, data, config, generator):
    was_training = model.training
    model.eval()
    try:
        values = []
        with torch.inference_mode():
            for _ in range(config.eval_batches):
                inputs, targets = _batch(data, config.batch_size, model.config.block_size, generator)
                _, loss = model(inputs, targets)
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Validation produced a non-finite loss. Try a smaller learning rate.")
                values.append(float(loss.item()))
        return sum(values) / len(values)
    finally:
        model.train(was_training)


def train(corpus, model_config=None, training_config=None, progress=None, cancel_event=None):
    """Train a new local model; returned partial results preserve cancellation state.

    Corpus splitting is contiguous and is an execution/learning aid. The tiny
    demonstration corpus does not establish an independent quality benchmark.
    """
    if not isinstance(corpus, str):
        raise ValueError("The corpus must be UTF-8 text.")
    try:
        raw = corpus.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("The corpus contains text that cannot be encoded as UTF-8.") from exc
    if not raw or len(raw) > MAX_CORPUS_BYTES:
        raise ValueError("Choose a nonempty UTF-8 corpus no larger than 4 MiB.")
    config = training_config or TrainingConfig()
    if not isinstance(config, TrainingConfig):
        raise ValueError("training_config must be a validated TrainingConfig.")
    tokenizer = Tokenizer.from_text(corpus)
    architecture = model_config or replace(ModelConfig(), vocab_size=tokenizer.vocab_size)
    if not isinstance(architecture, ModelConfig):
        raise ValueError("model_config must be a validated ModelConfig.")
    if architecture.vocab_size != tokenizer.vocab_size:
        raise ValueError("Model vocabulary size must match the selected corpus tokenizer.")
    hidden_elements = (config.batch_size * architecture.block_size
                       * architecture.n_embd * architecture.n_layer)
    attention_elements = (config.batch_size * architecture.n_head
                          * architecture.block_size**2 * architecture.n_layer)
    if hidden_elements > 8_388_608 or attention_elements > 8_388_608:
        raise ValueError("This batch/model combination exceeds the desktop attention/activation budget. Reduce batch size, context, heads or layers.")
    split = int(config.train_fraction * len(corpus))
    if min(split, len(corpus) - split) <= architecture.block_size:
        raise ValueError("The corpus is too short for both training and validation contexts. Supply more text or a smaller context length.")
    tokens = torch.tensor(tokenizer.encode(corpus), dtype=torch.long)
    train_data, validation_data = tokens[:split], tokens[split:]

    # One thread is suitable for this small CPU model and keeps the UI responsive.
    # Do not set interop threads: callers or diagnostics may already configure it.
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    model = GPTLanguageModel(architecture).to("cpu")
    training_generator = torch.Generator(device="cpu").manual_seed(config.seed + 1)
    validation_generator = torch.Generator(device="cpu").manual_seed(config.seed + 2)
    result = TrainingResult(model=model, tokenizer=tokenizer, config=architecture,
                            training_config=config, steps_completed=0, history=[],
                            cancelled=False, corpus_sha256=hashlib.sha256(raw).hexdigest())
    if _cancelled(cancel_event):
        result.cancelled = True
        return result

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    started = time.monotonic()
    model.train()
    for step in range(1, config.steps + 1):
        if _cancelled(cancel_event):
            break
        inputs, targets = _batch(train_data, config.batch_size, architecture.block_size, training_generator)
        _, loss = model(inputs, targets)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Training produced a non-finite loss. Try a smaller learning rate.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        result.steps_completed = step
        if step == 1 or step % config.eval_every == 0 or step == config.steps:
            if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
                raise ValueError("Training produced non-finite weights. No model should be exported from this run.")
            metric = {"step": step, "total_steps": config.steps,
                      "train_loss": float(loss.item()),
                      "val_loss": _validation_loss(model, validation_data, config, validation_generator),
                      "elapsed_seconds": time.monotonic() - started}
            result.history.append(metric)
            if progress is not None:
                progress(dict(metric))
    result.cancelled = result.steps_completed < config.steps and _cancelled(cancel_event)
    return result


def generate(result, prompt="", max_new_tokens=128, temperature=1.0, top_k=None,
             seed=42, cancel_event=None):
    """Return prompt plus sampled characters, or a known-character start if empty."""
    if not isinstance(result, TrainingResult):
        raise ValueError("Load or train a model before generating text.")
    integer("max_new_tokens", max_new_tokens, 1, MAX_GENERATED_TOKENS)
    integer("seed", seed, 0, 2**31 - 1)
    number("temperature", temperature, 0.05, 5.0)
    if not isinstance(prompt, str) or len(prompt) > 4096:
        raise ValueError("The prompt must contain at most 4,096 characters.")
    if top_k is not None:
        integer("top_k", top_k, 1, result.tokenizer.vocab_size)
    encoded = result.tokenizer.encode(prompt) if prompt else [0]
    model = result.model
    device = next(model.parameters()).device
    if device.type == "cpu":
        torch.set_num_threads(result.training_config.threads)
    index = torch.tensor([encoded], dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                if _cancelled(cancel_event):
                    break
                logits, _ = model(index[:, -result.config.block_size:])
                logits = logits[:, -1, :] / temperature
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("The model produced non-finite values; use a valid saved model or retrain it.")
                if top_k is not None:
                    cutoff = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < cutoff, float("-inf"))
                probabilities = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probabilities, num_samples=1, generator=generator)
                index = torch.cat((index, next_id), dim=1)
        return result.tokenizer.decode(index[0].cpu().tolist())
    finally:
        model.train(was_training)
