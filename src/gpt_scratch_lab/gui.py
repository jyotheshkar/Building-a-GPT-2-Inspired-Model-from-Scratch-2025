"""Offline Tk desktop interface for the maintained GPT learning package.

Only the Tk thread touches widgets. Training, generation and checkpoint I/O
run in one background worker at a time and report through a queue.
"""
from __future__ import annotations

from importlib import resources
import math
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

BG = "#0d1117"
PANEL = "#161b22"
FIELD = "#0d1117"
TEXT = "#e6edf3"
MUTED = "#9198a1"
BLUE = "#58a6ff"
LINE = "#30363d"
MAX_CORPUS_BYTES = 4 * 1024 * 1024
PRESETS = ("Tiny CPU (recommended)", "Small CPU (more memory)")


class CoreBackend:
    """Small adapter that keeps heavy Torch imports off initial widget creation."""

    def demo(self) -> str:
        return resources.files("gpt_scratch_lab").joinpath("data/demo.txt").read_text(encoding="utf-8")

    def train(self, corpus: str, settings: dict, cancel: threading.Event, progress: Callable) -> Any:
        from .config import ModelConfig, TrainingConfig
        from .tokenizer import Tokenizer
        from .training import train

        model_config = None
        if settings["preset"] == PRESETS[1]:
            model_config = ModelConfig(vocab_size=Tokenizer.from_text(corpus).vocab_size,
                                       block_size=128, n_embd=128, n_head=4, n_layer=4)
        config = TrainingConfig(steps=settings["steps"], learning_rate=settings["learning_rate"],
                                seed=settings["seed"], batch_size=8, threads=1)
        return train(corpus, model_config=model_config, training_config=config,
                     progress=progress, cancel_event=cancel)

    def generate(self, result: Any, prompt: str, settings: dict, cancel: threading.Event) -> str:
        from .training import generate
        return generate(result, prompt=prompt, max_new_tokens=settings["characters"],
                        temperature=settings["temperature"], seed=settings["seed"], cancel_event=cancel)

    def save(self, path: str, result: Any) -> Any:
        from .checkpoint import save_checkpoint
        return save_checkpoint(path, result)

    def load(self, path: str) -> Any:
        from .checkpoint import load_checkpoint
        return load_checkpoint(path)

    def describe(self, result: Any) -> str:
        parameters = sum(parameter.numel() for parameter in result.model.parameters())
        state = "partial / cancelled" if result.cancelled else "completed"
        return (f"{parameters:,} parameters · {result.tokenizer.vocab_size} characters · "
                f"{result.steps_completed:,} training steps ({state})\n"
                f"{result.config.n_layer} blocks · {result.config.n_embd} dimensions · "
                f"{result.config.block_size}-character context · CPU\n"
                f"Corpus SHA-256: {result.corpus_sha256}")

    def vocabulary(self, result: Any) -> tuple[str, ...]:
        return tuple(result.tokenizer.chars)


def _integer(value: str, name: str, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a whole number.") from exc
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} must be between {minimum:,} and {maximum:,}.")
    return number


def _decimal(value: str, name: str, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be finite and between {minimum:g} and {maximum:g}.")
    return number


def _loss_history(history: Any) -> list[tuple[int, float, float | None]]:
    """Validate the display boundary even when a core/backend result is faulty."""
    if not isinstance(history,list):
        raise ValueError("Model loss history must be a list.")
    result=[]
    for row in history:
        if not isinstance(row,dict) or type(row.get("step")) is not int or row["step"]<0:
            raise ValueError("Model loss history contains an invalid step.")
        train_loss=row.get("train_loss")
        val_loss=row.get("val_loss")
        if (isinstance(train_loss,bool) or not isinstance(train_loss,(int,float))
                or not math.isfinite(train_loss)):
            raise ValueError("Model loss history contains an invalid training loss.")
        if val_loss is not None and (isinstance(val_loss,bool)
                or not isinstance(val_loss,(int,float)) or not math.isfinite(val_loss)):
            raise ValueError("Model loss history contains an invalid validation loss.")
        result.append((row["step"],float(train_loss),None if val_loss is None else float(val_loss)))
    return result


class GPTScratchApp:
    def __init__(self, root: tk.Tk, backend: Any | None = None):
        self.root = root
        self.backend = backend if backend is not None else CoreBackend()
        self.result = None
        self.corpus = ""
        self.busy = False
        self.closing = False
        self._close_complete = False
        self._worker: threading.Thread | None = None
        self._cancel = threading.Event()
        self._events: queue.Queue = queue.Queue()
        self._poll_id: str | None = None
        self._losses: list[tuple[int, float, float | None]] = []
        self._controls: list[tuple[Any, str]] = []

        root.title("GPT From Scratch Lab — GPT-2-inspired learning")
        width = min(1080, max(880, root.winfo_screenwidth()-80))
        height = min(740, max(650, root.winfo_screenheight()-100))
        root.geometry(f"{width}x{height}")
        root.minsize(880, 650)
        root.configure(background=BG)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._style()
        self._build()
        self._use_demo(show_error=False)
        self._refresh_buttons()
        self._poll_id = root.after(75, self._drain_events)

    def _style(self):
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("Muted.TLabel", foreground=MUTED)
        style.configure("Title.TLabel", font=("Arial", 20, "bold"))
        style.configure("TLabelframe", background=BG, bordercolor=LINE)
        style.configure("TLabelframe.Label", background=BG, foreground=TEXT, font=("Arial", 10, "bold"))
        style.configure("TEntry", fieldbackground=FIELD, foreground=TEXT, insertcolor=TEXT, bordercolor=LINE)
        style.configure("TCombobox", fieldbackground=FIELD, background=PANEL, foreground=TEXT,
                        arrowcolor=MUTED, bordercolor=LINE)
        style.map("TCombobox", fieldbackground=[("readonly", FIELD)],
                  foreground=[("readonly", TEXT), ("disabled", MUTED)])
        style.configure("TButton", background=PANEL, foreground=TEXT, bordercolor=LINE, padding=(9, 6))
        style.map("TButton", background=[("active", "#24344a")], foreground=[("disabled", "#5c6673")])
        style.configure("Primary.TButton", background="#1f6feb", foreground="#ffffff")
        style.map("Primary.TButton", background=[("active", "#388bfd"), ("disabled", PANEL)])
        style.configure("Horizontal.TProgressbar", background=BLUE, troughcolor=PANEL, bordercolor=LINE)

    def _control(self, widget, idle_state="normal"):
        self._controls.append((widget, idle_state))
        return widget

    def _text(self, parent, height, editable=True):
        container = ttk.Frame(parent)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)
        widget = tk.Text(container, height=height, width=1, background=FIELD, foreground=TEXT,
                         insertbackground=TEXT, selectbackground="#264f78", wrap="word",
                         relief="flat", borderwidth=1, highlightthickness=1,
                         highlightbackground=LINE, padx=9, pady=8, font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=widget.yview)
        widget.configure(yscrollcommand=scrollbar.set)
        widget.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        if not editable:
            widget.configure(state="disabled")
        return container, widget

    def _build(self):
        outer = ttk.Frame(self.root, padding=18)
        outer.grid(sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=0, minsize=330)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(2, weight=1)
        ttk.Label(outer, text="GPT From Scratch Lab", style="Title.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(outer, text="Train a small character model, inspect its progress, and generate locally. No API key required.",
                  style="Muted.TLabel").grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 15))

        left = ttk.Frame(outer)
        left.grid(row=2, column=0, sticky="nsew", padx=(0, 16))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        source = ttk.LabelFrame(left, text="1  Text source", padding=10)
        source.grid(row=0, column=0, sticky="nsew")
        source.columnconfigure(0, weight=1)
        source.rowconfigure(3, weight=1)
        self.source_name = tk.StringVar(value="No text selected")
        ttk.Label(source, textvariable=self.source_name, wraplength=290).grid(row=0, column=0, sticky="w")
        actions = ttk.Frame(source)
        actions.grid(row=1, column=0, sticky="ew", pady=(8, 7))
        self.open_text_button = self._control(ttk.Button(actions, text="Load UTF-8 text", command=self._open_text))
        self.open_text_button.pack(side="left")
        self.demo_button = self._control(ttk.Button(actions, text="Use demo", command=self._use_demo))
        self.demo_button.pack(side="left", padx=(7, 0))
        self.source_stats = tk.StringVar(value="")
        ttk.Label(source, textvariable=self.source_stats, style="Muted.TLabel").grid(row=2, column=0, sticky="w", pady=(0, 6))
        preview_frame, self.preview = self._text(source, 7, editable=False)
        preview_frame.grid(row=3, column=0, sticky="nsew")
        ttk.Label(source, text="The synthetic demo tests execution. Use your own text for meaningful experiments.",
                  style="Muted.TLabel", wraplength=290).grid(row=4, column=0, sticky="w", pady=(7, 0))

        settings = ttk.LabelFrame(left, text="2  Train a fresh model", padding=10)
        settings.grid(row=1, column=0, sticky="ew", pady=(13, 0))
        settings.columnconfigure(1, weight=1)
        self.preset = tk.StringVar(value=PRESETS[0])
        ttk.Label(settings, text="Preset").grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.preset_box = self._control(ttk.Combobox(settings, textvariable=self.preset, values=PRESETS,
                                                    state="readonly", width=24), "readonly")
        self.preset_box.grid(row=0, column=1, sticky="ew", pady=3)
        self.preset_box.bind("<<ComboboxSelected>>", lambda _event: self._preset_description())
        self.steps = tk.StringVar(value="100")
        self.learning_rate = tk.StringVar(value="0.0003")
        self.seed = tk.StringVar(value="42")
        for row, (label, variable) in enumerate((("Steps", self.steps), ("Learning rate", self.learning_rate), ("Seed", self.seed)), 1):
            ttk.Label(settings, text=label).grid(row=row, column=0, sticky="w")
            entry = self._control(ttk.Entry(settings, textvariable=variable, width=20))
            entry.grid(row=row, column=1, sticky="ew", pady=3)
        self.preset_note = tk.StringVar()
        self._preset_description()
        ttk.Label(settings, textvariable=self.preset_note, style="Muted.TLabel", wraplength=290).grid(row=4, column=0, columnspan=2, sticky="w", pady=(7, 5))
        train_actions = ttk.Frame(settings)
        train_actions.grid(row=5, column=0, columnspan=2, sticky="ew")
        self.train_button = ttk.Button(train_actions, text="Train model", command=self._start_training, style="Primary.TButton")
        self.train_button.pack(side="left", expand=True, fill="x")
        self.cancel_button = ttk.Button(train_actions, text="Stop", command=self._stop, state="disabled")
        self.cancel_button.pack(side="left", padx=(7, 0))
        ttk.Label(settings, text="Training starts a new model. Loaded checkpoints are used for generation.",
                  style="Muted.TLabel", wraplength=290).grid(row=6, column=0, columnspan=2, sticky="w", pady=(7, 0))

        right = ttk.Frame(outer)
        right.grid(row=2, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        progress = ttk.LabelFrame(right, text="Training progress & current model", padding=10)
        progress.grid(row=0, column=0, sticky="ew")
        progress.columnconfigure(0, weight=1)
        self.progress_note = tk.StringVar(value="Ready. Train from text or load a trusted checkpoint.")
        progress_label = ttk.Label(progress, textvariable=self.progress_note, wraplength=620)
        progress_label.grid(row=0, column=0, sticky="ew")
        self.progress_bar = ttk.Progressbar(progress, maximum=100)
        self.progress_bar.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        self.loss_canvas = tk.Canvas(progress, height=92, background=FIELD, highlightthickness=1, highlightbackground=LINE)
        self.loss_canvas.grid(row=2, column=0, sticky="ew")
        self.loss_canvas.bind("<Configure>", lambda _event: self._draw_loss())
        self.model_details = tk.StringVar(value="No model in memory. No pretrained weights are supplied.")
        model_label = ttk.Label(progress, textvariable=self.model_details, style="Muted.TLabel", wraplength=620)
        model_label.grid(row=3, column=0, sticky="ew", pady=(8, 7))
        cp_actions = ttk.Frame(progress)
        cp_actions.grid(row=4, column=0, sticky="ew")
        self.load_checkpoint_button = self._control(ttk.Button(cp_actions, text="Load checkpoint", command=self._load_checkpoint))
        self.load_checkpoint_button.pack(side="left")
        self.save_checkpoint_button = ttk.Button(cp_actions, text="Save checkpoint", command=self._save_checkpoint, state="disabled")
        self.save_checkpoint_button.pack(side="left", padx=(7, 0))
        ttk.Label(progress, text="Load only checkpoints you created or trust.", style="Muted.TLabel").grid(row=5, column=0, sticky="w", pady=(6, 0))

        generation = ttk.LabelFrame(right, text="3  Generate text", padding=10)
        generation.grid(row=1, column=0, sticky="nsew", pady=(13, 0))
        generation.columnconfigure(0, weight=1)
        generation.rowconfigure(4, weight=1)
        ttk.Label(generation, text="Prompt (use characters present in the model's vocabulary)").grid(row=0, column=0, sticky="w")
        prompt_frame, self.prompt = self._text(generation, 2)
        self._control(self.prompt)
        prompt_frame.grid(row=1, column=0, sticky="ew", pady=(5, 7))
        options = ttk.Frame(generation)
        options.grid(row=2, column=0, sticky="ew")
        self.characters = tk.StringVar(value="160")
        self.temperature = tk.StringVar(value="0.8")
        ttk.Label(options, text="New characters").pack(side="left")
        self._control(ttk.Entry(options, textvariable=self.characters, width=6)).pack(side="left", padx=(6, 12))
        ttk.Label(options, text="Temperature").pack(side="left")
        self._control(ttk.Entry(options, textvariable=self.temperature, width=6)).pack(side="left", padx=(6, 10))
        self.generate_button = ttk.Button(options, text="Generate", command=self._generate, style="Primary.TButton", state="disabled")
        self.generate_button.pack(side="right")
        output_note = ttk.Label(generation, text="Output includes the prompt. Short training runs are educational, not a language-quality benchmark.",
                                style="Muted.TLabel", wraplength=620)
        output_note.grid(row=3, column=0, sticky="ew", pady=(8, 5))
        output_frame, self.output = self._text(generation, 5, editable=False)
        output_frame.grid(row=4, column=0, sticky="nsew")
        self.export_button = ttk.Button(generation, text="Save text…", command=self._save_text, state="disabled")
        self.export_button.grid(row=5, column=0, sticky="e", pady=(7, 0))
        self.status = tk.StringVar(value="Offline · CPU · no API key")
        ttk.Label(outer, textvariable=self.status, style="Muted.TLabel").grid(row=3, column=0, columnspan=2, sticky="w", pady=(12, 0))
        progress.bind("<Configure>", lambda event:[label.configure(wraplength=max(250,event.width-24))
                                                   for label in (progress_label,model_label)])
        generation.bind("<Configure>", lambda event:output_note.configure(wraplength=max(250,event.width-24)))
        self._draw_loss()

    def _preset_description(self):
        small = self.preset.get() == PRESETS[1]
        self.preset_note.set("CPU: 4 blocks · 128 dimensions · 128-character context." if small else
                             "CPU: 2 blocks · 64 dimensions · 64-character context.")

    @staticmethod
    def _replace_text(widget: tk.Text, value: str, disabled=True):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        if disabled:
            widget.configure(state="disabled")

    def _set_source(self, value: str, name: str):
        if not value.strip():
            raise ValueError("Choose a text file containing more than whitespace.")
        self.corpus = value
        self.source_name.set(name)
        self.source_stats.set(f"{len(value):,} characters · {len(set(value)):,} unique characters")
        preview = value[:2500] + ("\n… preview truncated" if len(value)>2500 else "")
        self._replace_text(self.preview, preview)
        self.status.set("Text loaded locally. Training will start a fresh model.")
        self._refresh_buttons()

    def _use_demo(self, show_error=True):
        if self.busy:
            return
        try:
            self._set_source(self.backend.demo(), "Synthetic demo corpus")
        except Exception as exc:
            self.status.set("Demo unavailable. Load a UTF-8 text file.")
            if show_error:
                self._error(exc)

    def _open_text(self):
        if self.busy:
            return
        name = filedialog.askopenfilename(parent=self.root, title="Choose local UTF-8 training text",
                                         filetypes=[("Text files", "*.txt"), ("All files", "*")])
        if not name:
            return
        try:
            path = Path(name)
            if path.stat().st_size > MAX_CORPUS_BYTES:
                raise ValueError("For this laptop interface, use a UTF-8 text file no larger than 4 MiB.")
            self._set_source(path.read_text(encoding="utf-8-sig"), path.name)
        except UnicodeError:
            self._error(ValueError("The selected file is not UTF-8 text. Convert it to UTF-8, then try again."))
        except Exception as exc:
            self._error(exc)

    def _settings(self) -> dict:
        if self.preset.get() not in PRESETS:
            raise ValueError("Choose one of the CPU presets.")
        return {"preset":self.preset.get(), "steps":_integer(self.steps.get(), "Steps", 1, 10_000),
                "learning_rate":_decimal(self.learning_rate.get(), "Learning rate", 1e-6, 0.1),
                "seed":_integer(self.seed.get(), "Seed", 0, 2**31-1)}

    def _start_training(self):
        if self.busy:
            return
        try:
            settings = self._settings()
            if not self.corpus:
                raise ValueError("Load a text source before training.")
        except ValueError as exc:
            self._error(exc)
            return
        corpus = self.corpus
        self._losses = []
        self._draw_loss()
        self.progress_bar.configure(maximum=settings["steps"], value=0)
        self.progress_note.set("Starting a fresh model on CPU…")
        self._submit("train", lambda cancel: self.backend.train(corpus, settings, cancel,
                     lambda event: self._events.put(("progress", event))))

    def _generate(self):
        if self.busy or self.result is None:
            return
        try:
            settings = {"characters":_integer(self.characters.get(), "New characters", 1, 1000),
                        "temperature":_decimal(self.temperature.get(), "Temperature", 0.05, 2.0),
                        "seed":_integer(self.seed.get(), "Seed", 0, 2**31-1)}
            prompt = self.prompt.get("1.0", "end-1c")
        except ValueError as exc:
            self._error(exc)
            return
        result = self.result
        self._submit("generate", lambda cancel: self.backend.generate(result, prompt, settings, cancel))

    def _save_checkpoint(self):
        if self.busy or self.result is None:
            return
        path = filedialog.asksaveasfilename(parent=self.root, title="Save model, vocabulary and configuration",
                                           defaultextension=".pt", initialfile="gpt-lab-checkpoint.pt",
                                           filetypes=[("GPT Lab checkpoint", "*.pt")])
        if path:
            result = self.result
            self._submit("save", lambda _cancel: self.backend.save(path, result))

    def _load_checkpoint(self):
        if self.busy:
            return
        path = filedialog.askopenfilename(parent=self.root, title="Open a checkpoint you created or trust",
                                         filetypes=[("GPT Lab checkpoint", "*.pt"), ("All files", "*")])
        if path:
            self._submit("load", lambda _cancel: self.backend.load(path))

    def _save_text(self):
        if self.busy:
            return
        value = self.output.get("1.0", "end-1c")
        if not value:
            return
        path = filedialog.asksaveasfilename(parent=self.root, title="Save generated text",
                                           defaultextension=".txt", initialfile="generated-text.txt",
                                           filetypes=[("UTF-8 text", "*.txt")])
        if path:
            def export(_cancel):
                Path(path).write_text(value, encoding="utf-8")
                return path
            self._submit("export", export)

    def _submit(self, kind: str, operation: Callable):
        if self.busy or self.closing:
            return
        self._cancel = threading.Event()
        cancel = self._cancel
        self.busy = True
        self._busy_kind = kind
        self.status.set({"train":"Training locally on CPU…", "generate":"Generating locally…",
                         "save":"Saving checkpoint…", "load":"Loading checkpoint…",
                         "export":"Saving text…"}[kind])
        self._refresh_buttons()
        def worker():
            try:
                value = operation(cancel)
                self._events.put(("done", (kind, value, None)))
            except Exception as exc:
                self._events.put(("done", (kind, None, str(exc) or type(exc).__name__)))
        self._worker = threading.Thread(target=worker, name="gpt-lab-worker", daemon=True)
        self._worker.start()

    def _stop(self):
        if self.busy:
            self._cancel.set()
            self.cancel_button.configure(state="disabled")
            self.status.set("Stopping after the current computation…")

    def _refresh_buttons(self):
        for widget, idle in self._controls:
            widget.configure(state="disabled" if self.busy else idle)
        self.train_button.configure(state="normal" if self.corpus and not self.busy else "disabled")
        self.cancel_button.configure(state="normal" if self.busy and getattr(self,"_busy_kind","") in ("train","generate") and not self._cancel.is_set() else "disabled")
        for widget in (self.generate_button, self.save_checkpoint_button):
            widget.configure(state="normal" if self.result is not None and not self.busy else "disabled")
        has_output = bool(self.output.get("1.0", "end-1c"))
        self.export_button.configure(state="normal" if has_output and not self.busy else "disabled")

    def _drain_events(self):
        if self.closing:
            return
        try:
            while True:
                try:
                    kind, value = self._events.get_nowait()
                except queue.Empty:
                    break
                try:
                    if kind == "progress":
                        self._progress(value)
                    elif kind == "done":
                        self._completed(*value)
                    else:
                        raise ValueError("Unexpected desktop worker event.")
                except Exception as exc:
                    self._cancel.set()
                    self.status.set("Invalid operation data; stopping the current work.")
                    self._error(ValueError(f"Could not update the desktop: {exc}"))
        finally:
            # A malformed metric must not permanently stop Tk's queue pump.
            if not self.closing:
                self._poll_id = self.root.after(75, self._drain_events)

    def _progress(self, event):
        if not isinstance(event,dict) or type(event.get("total_steps")) is not int or event["total_steps"]<1:
            raise ValueError("Training progress contains an invalid total.")
        step,train_loss,val_loss = _loss_history([event])[0]
        total=event["total_steps"]
        if step>total:
            raise ValueError("Training progress exceeds its configured step count.")
        validation = "" if val_loss is None else f" · validation loss {float(val_loss):.4f}"
        elapsed = event.get("elapsed_seconds",0)
        if isinstance(elapsed,bool) or not isinstance(elapsed,(int,float)) or not math.isfinite(elapsed) or elapsed<0:
            raise ValueError("Training progress contains an invalid elapsed time.")
        self.progress_note.set(f"Step {step:,} / {total:,} · train loss {train_loss:.4f}{validation} · {elapsed:.1f}s")
        self.progress_bar.configure(maximum=total, value=step)
        self._losses.append((step,train_loss,None if val_loss is None else float(val_loss)))
        self._draw_loss()

    def _completed(self, kind: str, value: Any, error: str | None):
        self.busy = False
        self._worker = None
        if error is not None:
            self.status.set("Operation failed; the previous model and output are retained.")
            self._refresh_buttons()
            self._error(ValueError(error))
            return
        if kind in ("train","load"):
            try:
                details=self.backend.describe(value)
                chars=self.backend.vocabulary(value)
                if not chars or any(not isinstance(char,str) or len(char)!=1 for char in chars):
                    raise ValueError("Model vocabulary is invalid.")
                if type(value.steps_completed) is not int or value.steps_completed<0:
                    raise ValueError("Model training step count is invalid.")
                history=_loss_history(value.history)
            except Exception as exc:
                self.status.set("Invalid model metadata; the previous model is retained.")
                self._refresh_buttons()
                self._error(ValueError(f"Could not display this model: {exc}"))
                return
            prompt = self.prompt.get("1.0", "end-1c")
            self.result = value
            self.model_details.set(details)
            if not prompt or any(char not in chars for char in prompt):
                preferred = self.corpus[:48] if kind=="train" else next((char for char in chars if not char.isspace()),chars[0])
                self._replace_text(self.prompt,preferred,disabled=False)
            if kind=="load":
                self._losses = history
                self._draw_loss()
                self.progress_note.set("Checkpoint loaded for generation. Training starts a fresh model.")
            else:
                partial = "Stopped" if value.cancelled else "Finished"
                self.progress_note.set(f"{partial} after {value.steps_completed:,} training steps. Model is ready for generation.")
            self.status.set("Model ready locally. Save a checkpoint to keep it after closing.")
        elif kind=="generate":
            self._replace_text(self.output,value)
            self.status.set("Generation stopped; partial text retained." if self._cancel.is_set() else "Generated locally. Output includes the prompt.")
        else:
            self.status.set(f"Saved: {Path(value).name}")
        self._refresh_buttons()

    def _draw_loss(self):
        canvas = self.loss_canvas
        canvas.delete("all")
        width, height = max(canvas.winfo_width(),300), 92
        canvas.create_text(9,11,anchor="w",text="TRAIN",fill=BLUE,font=("Arial",8))
        canvas.create_text(66,11,anchor="w",text="VALIDATION",fill="#e3b86c",font=("Arial",8))
        if not self._losses:
            canvas.create_text(width/2,52,text="Loss history will appear during training",fill=MUTED,font=("Arial",9))
            return
        values = [loss for _,train_loss,val_loss in self._losses for loss in (train_loss,val_loss)
                  if loss is not None and math.isfinite(loss)]
        if not values:
            return
        low, high = min(values), max(values)
        span = max(high-low,0.1)
        max_step = max(row[0] for row in self._losses)
        for offset,color in ((1,BLUE),(2,"#e3b86c")):
            coordinates=[]
            for row in self._losses:
                loss = row[offset]
                if loss is None or not math.isfinite(loss):
                    continue
                coordinates.extend((12+(width-24)*row[0]/max(1,max_step), height-12-(height-36)*(loss-low)/span))
            if len(coordinates)>=4:
                canvas.create_line(*coordinates,fill=color,width=2)
            elif len(coordinates)==2:
                x,y=coordinates
                canvas.create_oval(x-2,y-2,x+2,y+2,fill=color,outline=color)

    def _error(self, error: Exception):
        if not self.closing:
            messagebox.showerror("GPT From Scratch Lab",str(error),parent=self.root)

    def close(self):
        if self.closing:
            return
        self.closing = True
        self._cancel.set()
        if self._poll_id is not None:
            try:
                self.root.after_cancel(self._poll_id)
            except tk.TclError:
                pass
        # Hide immediately, then leave the event loop alive until native compute
        # or atomic checkpoint I/O finishes. Never join a worker on the Tk thread.
        self.root.withdraw()
        self._finish_close()

    def _finish_close(self):
        if self._worker is not None and self._worker.is_alive():
            self.root.after(25,self._finish_close)
        else:
            self._close_complete=True
            self.root.destroy()


def build_app(root: tk.Tk) -> GPTScratchApp:
    return GPTScratchApp(root)


def smoke_gui(run_compute: bool = True) -> dict:
    """Exercise real widgets and their core adapters in a bounded withdrawn UI.

    The default release check trains, generates and round-trips a checkpoint
    through the actual button handlers. ``run_compute=False`` is only the
    dependency-light widget construction check used during local UI development.
    """
    import tempfile
    import time

    root = tk.Tk()
    root.withdraw()
    app = None
    try:
        app = build_app(root)
        root.update_idletasks()
        root.update()
        widgets = all(widget.winfo_exists() for widget in
                      (app.train_button,app.generate_button,app.prompt,app.output,app.preview,app.loss_canvas))
        idle = (not app.busy and str(app.train_button.cget("state"))=="normal"
                and str(app.generate_button.cget("state"))=="disabled"
                and str(app.cancel_button.cget("state"))=="disabled")
        loaded = bool(app.corpus.strip())
        passed = bool(widgets and idle and loaded)
        report = {"status":"passed" if passed else "failed", "widgets_constructed":bool(widgets),
                  "default_source_loaded":loaded, "initial_idle_state":idle,
                  "tk_version":str(root.tk.call("package","present","Tk")),
                  "tcl_version":str(root.tk.call("info","patchlevel")),
                  "compute_exercised":bool(run_compute)}
        if not passed:
            raise RuntimeError(f"Desktop widget smoke failed: {report}")
        if run_compute:
            errors=[]
            app._error=lambda error:errors.append(str(error))

            def finish_operation():
                deadline=time.monotonic()+30
                while app.busy:
                    if time.monotonic()>deadline:
                        app._stop()
                        raise TimeoutError("Desktop smoke operation exceeded 30 seconds")
                    root.update()
                    time.sleep(0.005)
                root.update()
                if errors:
                    raise RuntimeError("Desktop adapter failed: "+errors[0])

            app.steps.set("2")
            app.characters.set("8")
            app.seed.set("42")
            app.train_button.invoke()
            finish_operation()
            if app.result is None or app.result.steps_completed!=2 or app.result.cancelled:
                raise RuntimeError("Desktop training handler did not complete two real updates")
            report["gui_training_steps"]=app.result.steps_completed
            app.generate_button.invoke()
            finish_operation()
            generated=app.output.get("1.0","end-1c")
            prompt=app.prompt.get("1.0","end-1c")
            if not generated.startswith(prompt) or len(generated)!=len(prompt)+8:
                raise RuntimeError("Desktop generation handler did not return eight new characters")
            report["gui_generation_length"]=len(generated)

            # File-picker selections are controlled fixture paths. All compute,
            # serialization and UI event handlers remain the production paths.
            with tempfile.TemporaryDirectory(prefix="gpt-lab-gui-") as temporary:
                checkpoint=Path(temporary)/"desktop-smoke.pt"
                text_file=Path(temporary)/"desktop-smoke.txt"
                previous_save=filedialog.asksaveasfilename
                previous_open=filedialog.askopenfilename
                try:
                    filedialog.asksaveasfilename=lambda **_kwargs:str(checkpoint)
                    app.save_checkpoint_button.invoke()
                    finish_operation()
                    if not checkpoint.is_file() or checkpoint.stat().st_size==0:
                        raise RuntimeError("Desktop checkpoint save did not create a file")
                    expected_vocabulary=app.backend.vocabulary(app.result)
                    expected_corpus=app.result.corpus_sha256
                    app.result=None
                    app._refresh_buttons()
                    filedialog.askopenfilename=lambda **_kwargs:str(checkpoint)
                    app.load_checkpoint_button.invoke()
                    finish_operation()
                    if (app.result is None or app.result.steps_completed!=2
                            or app.backend.vocabulary(app.result)!=expected_vocabulary
                            or app.result.corpus_sha256!=expected_corpus):
                        raise RuntimeError("Desktop checkpoint handler did not restore model metadata")
                    app.generate_button.invoke()
                    finish_operation()
                    restored=app.output.get("1.0","end-1c")
                    if restored!=generated:
                        raise RuntimeError("Desktop checkpoint changed seeded generation")
                    report["gui_checkpoint_roundtrip"]=True
                    filedialog.asksaveasfilename=lambda **_kwargs:str(text_file)
                    app.export_button.invoke()
                    finish_operation()
                    if not text_file.is_file() or text_file.read_text(encoding="utf-8")!=generated:
                        raise RuntimeError("Desktop export handler did not preserve generated text")
                    report["gui_text_export"]=True
                finally:
                    filedialog.asksaveasfilename=previous_save
                    filedialog.askopenfilename=previous_open
            restored_idle=(not app.busy
                           and str(app.train_button.cget("state"))=="normal"
                           and str(app.generate_button.cget("state"))=="normal"
                           and str(app.cancel_button.cget("state"))=="disabled")
            if not restored_idle:
                raise RuntimeError("Desktop did not restore idle controls after operations")
            report["idle_controls_restored"]=True
        return report
    finally:
        if app is not None:
            app.close()
        else:
            root.destroy()


def main():
    root = tk.Tk()
    build_app(root)
    root.mainloop()


if __name__ == "__main__":
    main()
