# Download and run GPT Scratch Lab

Open [desktop release 1.0.0](https://github.com/jyotheshkar/Creating-a-Large-Language-Model-from-Scratch-2025/releases/tag/desktop-v1.0.0). Select a native application under **Assets** for your computer. The native package includes Python, CPU PyTorch, NumPy and the Tk desktop runtime; a separate Python installation or AI API key is not required.

## Choose your download

| Computer | Native asset | Open after extracting |
| --- | --- | --- |
| Windows 10/11, Intel/AMD 64-bit | `GPTScratchLab-windows-x64.zip` | `GPTScratchLab/GPTScratchLab.exe` |
| Ubuntu 22.04+, Intel/AMD 64-bit | `GPTScratchLab-linux-x64.tar.gz` | `GPTScratchLab/GPTScratchLab` |
| macOS 15+, Apple silicon | `GPTScratchLab-macos-arm64.zip` | `GPTScratchLab/GPTScratchLab.app` |

These are native build targets. Each archive is extracted and its actual executable tested on its corresponding runner before publication. The automated GUI check constructs and updates a withdrawn Tk window; it does not establish visible-window compatibility on every laptop. Intel Mac, Windows ARM64 and Linux ARM64 native packages are not supplied in this release.

The release also includes a platform-independent Python wheel (`.whl`) and source distribution (`.tar.gz`) for developers. Those need Python and installed dependencies. GitHub's automatically generated **Source code** archives include the repository's historical materials; they are different from the scoped maintained Python source distribution and native application downloads.

## Windows

1. Download `GPTScratchLab-windows-x64.zip` and `SHA256SUMS.txt` from the same release.
2. Compare the archive's SHA-256 with the matching checksum line, as explained below.
3. Use **Extract All**, open the extracted `GPTScratchLab` folder, and open `GPTScratchLab.exe`.
4. Keep `_internal` and all bundled libraries beside the executable. Launching from inside the ZIP or moving only the executable will break the package.

This is a portable application and does not need an installer or administrator access. It has no publisher certificate, so Windows may show an unknown-publisher warning. Verify the download's origin and checksum; do not disable Windows security protections.

## Linux

From a graphical Ubuntu 22.04+ x64 desktop, download the Linux archive and checksum file, then run:

```sh
sha256sum --ignore-missing -c SHA256SUMS.txt
tar -xzf GPTScratchLab-linux-x64.tar.gz
./GPTScratchLab/GPTScratchLab
```

Continue only when your downloaded archive reports `OK`. Extract onto a filesystem that preserves executable permissions and symbolic links, and keep the full folder intact. Older GNU C libraries, Alpine/musl and other distributions are outside the native build baseline. A normal desktop display is needed for the visible application.

## macOS

1. Download `GPTScratchLab-macos-arm64.zip` and `SHA256SUMS.txt` from the same release.
2. Verify the ZIP's checksum, then extract it with Finder's Archive Utility.
3. Open `GPTScratchLab/GPTScratchLab.app` on an Apple-silicon Mac running macOS 15 or newer.

The application has PyInstaller's local ad-hoc signature, without Apple Developer ID verification or notarization. Gatekeeper may prevent a downloaded copy from opening. A checksum confirms that the file matches this release; it does not establish a verified publisher. See [Apple's downloaded-app guidance](https://support.apple.com/102445). No command to disable those checks is required or provided here.

## First training session

Choose the bundled demonstration text or load your own UTF-8 text file. The app works offline. Start with its Tiny CPU defaults and a short run, observe training progress, and use **Cancel** to stop between optimizer updates. Larger datasets and longer runs take more time; no fixed training duration is promised.

After training, enter a prompt containing only characters present in the selected corpus. An empty prompt uses a known-vocabulary starting character. Generate a short sample, export the text if useful, and save a checkpoint for a later session. The checkpoint stores weights, vocabulary, architecture and training metadata. It does not include the original corpus or optimizer state; reopening supports generation rather than a promise of bit-identical resumed training.

The original synthetic demo is an execution fixture. A small local model can produce nonsense after a short run. Successful training updates, finite loss and generated characters are execution evidence, rather than proof of useful language understanding or benchmark quality.

## Python wheel or source checkout

Use Python 3.11 and a virtual environment. Install CPU PyTorch from its official index, then install the downloaded wheel:

```sh
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install ./PATH_TO_DOWNLOADED_WHEEL.whl
python -m gpt_scratch_lab --help
python -m gpt_scratch_lab gui
```

Replace the wheel placeholder with the actual filename listed in the release. To develop from a source checkout instead, use `python -m pip install .`. The package metadata installs its declared dependencies; the native build guide records the exact release environment.

A source Python installation must include Tkinter to open the GUI. Some Linux distributions provide it in a separate operating-system package. CLI training and generation do not require opening a Tk window. See `python -m gpt_scratch_lab train --help`, `generate --help`, and `demo --help` for the actual command arguments.

## Verify checksums

Download `SHA256SUMS.txt` from the same release. On Windows PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath .\GPTScratchLab-windows-x64.zip
Get-Content -LiteralPath .\SHA256SUMS.txt
```

On macOS:

```sh
shasum -a 256 GPTScratchLab-macos-arm64.zip
cat SHA256SUMS.txt
```

Compare the full hash for the exact archive filename. Hexadecimal letter case does not matter. If the hashes differ, download a fresh copy from the release before opening it.

## Recovery and checkpoint boundaries

| Symptom | Useful check |
| --- | --- |
| Application exits immediately or reports a missing DLL | Extract the entire archive again and retain `_internal`/all bundled libraries. Confirm the CPU architecture. |
| Linux cannot open a display | Launch from a graphical desktop; the GUI needs a display session. |
| macOS cannot verify the developer | Read the signing limitation above and Apple's guidance. A notarized distribution is not supplied. |
| A text file is rejected | Check UTF-8 encoding, available characters, file length and the app's displayed error. Tiny train/validation splits must each be long enough for a context window. |
| A prompt is rejected | Use characters found in the trained vocabulary, or start with an empty prompt. |
| A checkpoint is rejected | Check that it was saved by this application and fits its supported format/configuration limits. Historical notebook pickle files are not supported. |
| Output is repetitive or nonsensical | The bundled text and short defaults demonstrate execution. Try an appropriately licensed corpus and a considered training experiment rather than treating the fixture as a capable model. |

Open checkpoints you created or trust. The loader uses CPU-only restricted PyTorch loading and schema/resource checks, with no unrestricted-pickle fallback. Those checks do not make it a universal sandbox for hostile files. The app saves corpus hashes and model metadata in checkpoints, not the corpus itself. Exported text contains the prompt and generated text you choose to save.

When reporting a problem, include the release tag, asset filename, OS/CPU, exact error and whether the archive checksum matched. Remove personal text and checkpoints before attaching files publicly. Platform manifests identify the exact source commit and dependency versions.

## Rebuild and notices

See [BUILDING.md](https://github.com/jyotheshkar/Creating-a-Large-Language-Model-from-Scratch-2025/blob/desktop-v1.0.0/packaging/BUILDING.md) for native rebuilds. Keep bundled licence notices, dependency inventories and manifests with the application when redistributing it. The maintained package's MIT scope and the historical notebook boundaries are described in [LICENSE-SCOPE.md](https://github.com/jyotheshkar/Creating-a-Large-Language-Model-from-Scratch-2025/blob/desktop-v1.0.0/LICENSE-SCOPE.md).
