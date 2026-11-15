# Building a GPT-2-Inspired Model from Scratch

**GPT Scratch Lab** is an offline desktop application and Python package for learning how a small character-level GPT-style model trains and generates text. Load a UTF-8 corpus, train a tiny CPU model, inspect progress, generate characters, and save or reopen a checkpoint containing the model configuration, vocabulary and weights.

The maintained decoder uses GPT-2-inspired pre-layer normalization, causal attention, GELU blocks and learned positions. It has a small character vocabulary and laptop-sized defaults. It is an educational model trained locally, with no OpenAI API key or pretrained GPT-2 weights required.

## Download and use

Open [desktop release 1.0.0](https://github.com/jyotheshkar/Creating-a-Large-Language-Model-from-Scratch-2025/releases/tag/desktop-v1.0.0) and choose your computer's package:

| Computer | Native package |
| --- | --- |
| Windows 10/11 x64 | `GPTScratchLab-windows-x64.zip` |
| Ubuntu 22.04+ x64 | `GPTScratchLab-linux-x64.tar.gz` |
| macOS 15+ Apple silicon | `GPTScratchLab-macos-arm64.zip` |

Extract the complete archive and open the application inside its `GPTScratchLab` folder. The native packages include Python, CPU PyTorch and the desktop runtime. Follow the [download, launch and checksum instructions](docs/DOWNLOADS.md). Windows packages are unsigned; macOS packages have an ad-hoc signature without Developer ID or notarization.

The release also provides a Python wheel and source distribution for people who want to inspect or extend the implementation. These are GitHub release assets; publication to PyPI is not claimed.

## First session

1. Open the desktop application and select the bundled demonstration text or your own UTF-8 text file.
2. Start with the Tiny CPU settings and a short training run. Training progress appears while the window remains responsive; **Cancel** stops between updates.
3. Enter a prompt using characters present in the selected corpus, or leave it empty for a known-vocabulary start.
4. Generate text, save a checkpoint, or export the generated text. A saved checkpoint can be reopened in a later session.

The demonstration corpus is an original synthetic execution fixture. A short run proves that training and sampling execute; fluent writing, factual answers and language-model quality are not promised. Training a capable large model requires a different data and compute scale.

## Python quickstart

Use Python 3.11 and a virtual environment. On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\python.exe -m gpt_scratch_lab gui
```

On macOS/Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install .
.venv/bin/python -m gpt_scratch_lab gui
```

Source installations need a working Tkinter runtime for the GUI. The command line can be used without opening the desktop window:

```sh
python -m gpt_scratch_lab demo
python -m gpt_scratch_lab train --corpus my-text.txt --steps 100 --output my-model.pt
python -m gpt_scratch_lab generate my-model.pt --tokens 64
python -m gpt_scratch_lab --help
```

See the command-specific help for training on a text file, selecting an output checkpoint, and generating from a saved checkpoint. Downloaded wheel installation and the native application's requirements are explained in [DOWNLOADS.md](docs/DOWNLOADS.md).

## Learning sources and original work

- Andrej Karpathy: [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY), [lecture repository](https://github.com/karpathy/ng-video-lecture), and [nanoGPT](https://github.com/karpathy/nanoGPT).
- Direct historical notebook lineage: Elliot Arledge's [freeCodeCamp course](https://www.youtube.com/watch?v=UU1WVnMk4E8) and [source repository](https://github.com/Infatoshi/fcc-intro-to-llms).
- The retained [bigram notebook](bigram.ipynb) and [GPT study notebook](gpt-v1%20%283%29.ipynb) preserve the earlier learning path. Exact originals, including historical outputs, are archived with hashes under [legacy](legacy/notebooks-before-desktop-2026-11-15/).

[Learning evidence and attribution](docs/LEARNING_EVIDENCE.md) distinguishes source lineage, study references, retained artifacts and newly added release features. A repository or video reference is not a course-completion certificate.

## Engineering documentation

| Document | Purpose |
| --- | --- |
| [Architecture](docs/ARCHITECTURE.md) | Model/data flow, module boundaries and laptop limits |
| [Verification](docs/VERIFICATION.md) | Model, checkpoint, notebook and extracted-application checks |
| [Provenance](docs/PROVENANCE.md) | Source commits, preserved artifacts and dated maintenance |
| [Download guide](docs/DOWNLOADS.md) | Supported targets, launch, checksums and recovery |
| [Native build guide](packaging/BUILDING.md) | Build environment, notices and release procedure |
| [Licence scope](LICENSE-SCOPE.md) | Maintained package versus historical/third-party material |

The `2025` suffix records the repository's creation year, **20 January 2025**. Desktop/package engineering is dated **15 November 2026**.
