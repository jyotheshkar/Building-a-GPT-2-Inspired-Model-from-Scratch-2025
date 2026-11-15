"""Command-line interface for local training, generation and desktop launch."""
import argparse
import json
from pathlib import Path
import sys

from . import __version__


def _emit(value):
    if sys.stdout is not None:
        print(value, flush=True)


def _corpus(path):
    from .config import MAX_CORPUS_BYTES
    if path is None:
        from .training import load_demo_corpus
        return load_demo_corpus()
    file = Path(path).expanduser()
    if not file.is_file() or file.stat().st_size > MAX_CORPUS_BYTES:
        raise ValueError("Choose an existing UTF-8 text file no larger than 4 MiB.")
    try:
        return file.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("The corpus must be a UTF-8 plain-text file.") from exc


def _training_arguments(command):
    command.add_argument("--corpus", type=Path, help="Licensed UTF-8 text; default: bundled synthetic execution fixture")
    command.add_argument("--steps", type=int, default=100)
    command.add_argument("--batch-size", type=int, default=8)
    command.add_argument("--learning-rate", type=float, default=3e-4)
    command.add_argument("--context", type=int, default=64)
    command.add_argument("--embedding", type=int, default=64)
    command.add_argument("--heads", type=int, default=4)
    command.add_argument("--layers", type=int, default=2)
    command.add_argument("--dropout", type=float, default=0.1)
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--output", type=Path, default=Path("gpt-scratch-model.pt"))


def parser():
    argument_parser = argparse.ArgumentParser(description="GPT Scratch Lab: an offline GPT-2-inspired character-model learning app")
    argument_parser.add_argument("--version", action="version", version=f"GPT Scratch Lab {__version__}")
    argument_parser.add_argument("--smoke-test", action="store_true", help="Run a bounded training/checkpoint/generation/desktop execution check")
    argument_parser.add_argument("--smoke-report", type=Path, help="Write a JSON report; requires --smoke-test")
    commands = argument_parser.add_subparsers(dest="command")
    _training_arguments(commands.add_parser("train", help="Train a tiny local model and save weights plus vocabulary"))
    demo = commands.add_parser("demo", help="Train on the synthetic fixture, save, and generate a sample")
    _training_arguments(demo)
    demo.add_argument("--tokens", type=int, default=128)
    generate = commands.add_parser("generate", help="Generate from a checkpoint you created or trust")
    generate.add_argument("checkpoint", type=Path)
    generate.add_argument("--prompt", default="")
    generate.add_argument("--tokens", type=int, default=128)
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--top-k", type=int)
    generate.add_argument("--seed", type=int, default=42)
    commands.add_parser("gui", help="Open the desktop learning app")
    return argument_parser


def main(argv=None):
    argument_parser = parser()
    options = argument_parser.parse_args(argv)
    if options.smoke_report and not options.smoke_test:
        argument_parser.error("--smoke-report requires --smoke-test")
    if options.smoke_test:
        if options.command:
            argument_parser.error("Use --smoke-test without a subcommand")
        try:
            from .smoke import run_smoke
            report = run_smoke()
        except Exception as exc:
            # Windowed binaries may have no console; retain a machine-readable
            # failure instead of losing the error in a desktop traceback dialog.
            report = {"status": "failed", "version": __version__,
                      "frozen": bool(getattr(sys, "frozen", False)),
                      "device": "cpu", "error": str(exc)}
        if options.smoke_report:
            options.smoke_report.parent.mkdir(parents=True, exist_ok=True)
            options.smoke_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        _emit(json.dumps(report, sort_keys=True))
        return 0 if report["status"] == "passed" else 1
    if options.command == "gui" or (options.command is None and getattr(sys, "frozen", False)):
        from .gui import main as gui_main
        gui_main()
        return 0
    if options.command is None:
        argument_parser.print_help()
        return 0
    try:
        from .checkpoint import load_checkpoint, save_checkpoint
        from .config import ModelConfig, TrainingConfig
        from .tokenizer import Tokenizer
        from .training import generate, train
        if options.command in {"train", "demo"}:
            corpus = _corpus(options.corpus)
            tokenizer = Tokenizer.from_text(corpus)
            model_config = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=options.context,
                                       n_embd=options.embedding, n_head=options.heads,
                                       n_layer=options.layers, dropout=options.dropout)
            training_config = TrainingConfig(steps=options.steps, batch_size=options.batch_size,
                                             learning_rate=options.learning_rate, seed=options.seed)
            result = train(corpus, model_config, training_config,
                           progress=lambda metric: _emit(json.dumps(metric, sort_keys=True)))
            destination = save_checkpoint(options.output, result)
            _emit(f"Saved {destination} after {result.steps_completed} steps ({result.model.parameter_count:,} parameters).")
            if options.command == "demo":
                _emit(generate(result, max_new_tokens=options.tokens, seed=options.seed))
            return 0
        if options.command == "generate":
            result = load_checkpoint(options.checkpoint)
            _emit(generate(result, prompt=options.prompt, max_new_tokens=options.tokens,
                           temperature=options.temperature, top_k=options.top_k, seed=options.seed))
            return 0
    except (ValueError, OSError, RuntimeError) as exc:
        argument_parser.error(str(exc))
    return 0
