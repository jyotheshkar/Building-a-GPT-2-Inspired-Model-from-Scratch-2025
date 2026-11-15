# GPT Scratch Lab architecture

The maintained application is a local, CPU-first learning tool. The Tk desktop and command line share one tokenizer/model/training/checkpoint implementation. Input text, training and generation stay on the user's computer.

## Module boundaries

| Boundary | Source | Responsibility |
| --- | --- | --- |
| User interface | `src/gpt_scratch_lab/gui.py` | File selection, training settings, progress, cancellation, generation and explicit exports |
| Command line | `cli.py`, `__main__.py` | Validated commands, source/frozen launch and diagnostic mode |
| Configuration | `config.py` | Model/training limits and explicit CPU defaults |
| Tokenization | `tokenizer.py` | Character vocabulary, reversible IDs and clear unknown-character errors |
| Model | `model.py` | Token/position embeddings, causal decoder blocks, final normalization and logits |
| Training | `training.py` | Corpus splits, batches, AdamW updates, evaluation callbacks and cancellation |
| Persistence | `checkpoint.py` | Atomic save, bounded metadata, restricted CPU load and strict state validation |
| Verification | `smoke.py`, `tests/` | Actual optimizer/checkpoint/generation checks and boundary regressions |

## Model and the GPT-2-inspired name

The model is an autoregressive character decoder. A character ID selects a token embedding; learned position embeddings identify its context position. Each block applies layer normalization before causal self-attention and before a GELU feed-forward network, with residual connections around both. A final normalization and vocabulary projection produce next-character logits. Training compares those logits with the next-character targets using cross-entropy.

The causal mask prevents a token from attending to future positions. Generation crops the context, uses evaluation/inference mode and samples a next character from a temperature-controlled distribution. The vocabulary and model dimensions travel with the checkpoint so another laptop can reproduce the same input encoding.

The family reference is [OpenAI GPT-2's architecture](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/model.py). This release deliberately uses a tiny character vocabulary and smaller contexts/layers/widths. It does not implement GPT-2's full BPE tokenizer or claim compatibility with its released weights. Four layers in the old notebook do not mean GPT-4.

## Desktop concurrency

Training and generation run through worker tasks so Tk's event loop can continue updating. Worker results pass back through a queue; Tk widgets are updated on the main thread. State-changing controls are disabled while an operation is active. Cancellation is checked between bounded training/generation steps. Closing the window requests cancellation instead of blocking the interface on a long thread join.

One Torch computation thread is the default for the small CPU model. This avoids a large worker pool competing with the desktop event loop. Configuration and corpus sizes are bounded for the intended learning workflow; larger research jobs require a separately measured compute/data plan.

## Data and persistence

The bundled prose is an original synthetic demonstration fixture. Users can supply their own licensed UTF-8 text. A train/validation split is useful for observing the program, but a split of short or repetitive synthetic text is not an independent language-quality benchmark.

A checkpoint records a format version, model/training configuration, character vocabulary, CPU tensor state dictionary, completed-step/history metadata, cancellation state and corpus SHA-256. Corpus contents and optimizer state are not stored. Saving uses a temporary file and atomic replacement so a failed write does not deliberately replace a valid checkpoint with a partial file.

Loading uses `torch.load(..., map_location='cpu', weights_only=True)`, validates supported configuration/vocabulary/tensor contracts, and loads state strictly. There is no legacy unrestricted-pickle fallback. Resource caps and restricted loading are useful boundaries, rather than a claim that arbitrary hostile files are safe.

## Historical study implementation

The original notebooks use a different post-normalization/ReLU character model and a separate bigram baseline. Their exact earlier bytes and outputs are preserved under `legacy/notebooks-before-desktop-2026-11-15/`. Current notebook copies clear stale output and fix inference/default-prompt behavior while retaining that study architecture. The historical [diagram](../gpt%20architecture.png) and demonstration video describe that earlier work.

## Scaling decisions

The public API separates the model and training loop from Tk and the CLI, allowing experiments and scripts to reuse the same behavior. For larger runs, measure corpus diversity, tokenization, context/memory cost, validation design and compute before increasing configuration sizes. Distributed training, cloud serving and production availability are outside this desktop release's verified scope.
