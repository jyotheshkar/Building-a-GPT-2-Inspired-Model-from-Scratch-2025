# Build GPT Scratch Lab distributions

Use a fresh checkout of the release's source commit. Native builds must run on their actual operating system and architecture; PyInstaller does not cross-compile this application.

| Target | Runner | Application archive |
| --- | --- | --- |
| Windows x64 | `windows-2022` | `GPTScratchLab-windows-x64.zip` |
| Ubuntu 22.04 x64 | `ubuntu-22.04` | `GPTScratchLab-linux-x64.tar.gz` |
| macOS 15 Apple silicon | `macos-15` | `GPTScratchLab-macos-arm64.zip` |

Create a Python 3.11 virtual environment. Install the official CPU runtime and pinned build tools:

```sh
python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r packaging/requirements-build.txt
```

The build environment pins NumPy 2.4.6, PyInstaller 6.22.3, contrib hooks 2026.10, build 1.6.1, setuptools 84.2.1 and wheel 0.48.0. Transitive dependencies and actual Python patch versions are recorded in manifests; bit-for-bit reproducible builds are not claimed.

## Python wheel and source distribution

```sh
python -m unittest discover -s tests -v
python packaging/build_python.py --source-sha FULL_COMMIT_SHA
```

On Linux, run the GUI-dependent commands under a graphical desktop or Xvfb. The CI baseline installs `xvfb`, `xauth` and `tk8.6`, then prefixes the commands with:

```sh
xvfb-run -a -s '-screen 0 1280x1024x24' python packaging/build_python.py --source-sha FULL_COMMIT_SHA
```

The helper builds a source distribution, then a wheel **from that source distribution**. It refuses historical notebooks/media, checkpoints, links and unsafe member paths. It installs the wheel without replacing the separately installed CPU Torch runtime, clears `PYTHONPATH`, changes to an unrelated directory, and exercises training, checkpoint recovery, generation and a withdrawn Tk window. Python source distributions require Python and dependencies; native packages include their interpreter/runtime.

## Native bundles

Install the tested wheel, then choose only the matching target:

```sh
python packaging/install_wheel.py --source-sha FULL_COMMIT_SHA --assets release-assets
python packaging/build_desktop.py --target windows-x64 --source-sha FULL_COMMIT_SHA
python packaging/build_desktop.py --target linux-x64 --source-sha FULL_COMMIT_SHA
python packaging/build_desktop.py --target macos-arm64 --source-sha FULL_COMMIT_SHA
```

`FULL_COMMIT_SHA` is the exact forty-character source commit, not a branch name. The native helper refuses to reuse an existing `build/desktop/TARGET` directory. Linux needs the same graphical display/Xvfb wrapper as above. The complete portable directory is archived under `GPTScratchLab/`. The macOS ZIP contains `GPTScratchLab/GPTScratchLab.app`.

The spec keeps the standard Torch hook, including Python sources needed by optimizer introspection, and the standard `_tkinter` hook for Tcl/Tk scripts and libraries. Package data includes the offline demo corpus. Linux tar and macOS `ditto` preserve symlinks. Copying only the executable or flattening POSIX symlinks can break the runtime.

Each bundle is executed before archiving and again **after extracting its final archive** into another directory. The diagnostic performs real optimizer updates, checks finite/changed parameters and causality, saves and reloads a checkpoint/vocabulary, and generates text. It also constructs the actual Tk widgets and exercises their Train/Generate handlers, checkpoint save/load and text export using temporary paths. It does not measure language quality or replace human visible-window/laptop compatibility testing. macOS ad-hoc signatures are verified after extraction; no Developer ID, notarization or Windows publisher certificate is configured.

## Notices and provenance

Original notices from installed CPU PyTorch, NumPy, CPython, PyInstaller and hook distributions are copied into `licenses/`. Tcl and Tk are queried separately for their full runtime patch versions and script directories. Their actual local `license.terms` files are copied when present. When omitted from an interpreter distribution, the helper retrieves the original notice from that component's exact official `core-MAJOR-MINOR-PATCH` source tag, recording its URL and SHA-256. Moving branch fallbacks and assuming Tcl/Tk versions match are avoided. Bundled Tcl/Tk script headers remain unchanged. macOS system frameworks can remain OS-provided; manifests record actual runtime paths and whether that applies.

`LICENSE` and `LICENSE-SCOPE.md` apply only to the maintained code described there. Historical tutorial notebooks/media are preserved in Git history/repository archives and are not included in the Python distribution or assigned a new licence. Dependency components retain their own terms.

## Release lifecycle

Pull requests build and verify the Python package and all native targets with read-only repository permission. The publishing job alone requests `contents: write` on an exact `main` push in repository ID `919738589`. The repository name comes from the workflow environment, so an approved rename does not change that identity.

A new version creates and verifies its exact Git tag first, creates a draft with `--verify-tag`, uploads only the verified assets, compares their server-side digests and sizes, then publishes and reads back the result. Published releases are immutable. An interrupted draft or orphan tag fails safely for inspection. Later `main` pushes with the same version verify the published tag, manifests, archive digests and asset set, then skip rebuilding/publication. For another release, bump the package version and `packaging/release.json` version/tag/distribution filenames together. Existing release assets are never silently overwritten.
