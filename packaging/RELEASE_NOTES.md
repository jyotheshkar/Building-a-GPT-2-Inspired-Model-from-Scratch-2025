# GPT Scratch Lab desktop 1.0.0

A local educational character-language-model application for training a small GPT-2-inspired decoder, generating text and saving/reloading a model on a laptop. It runs offline without an account or API key. No pretrained GPT-2 weights or OpenAI model service is included.

| Asset | Use |
| --- | --- |
| `GPTScratchLab-windows-x64.zip` | Portable Windows x64 desktop application |
| `GPTScratchLab-linux-x64.tar.gz` | Portable Ubuntu 22.04+ x64 desktop application |
| `GPTScratchLab-macos-arm64.zip` | macOS 15+ Apple-silicon desktop application |
| `gpt_scratch_lab-1.0.0-py3-none-any.whl` | Installable maintained Python package; requires Python 3.11+ and CPU Torch |
| `gpt_scratch_lab-1.0.0.tar.gz` | Maintained source distribution; excludes historical tutorial notebooks/media |

Extract the whole native archive and open the application inside `GPTScratchLab/`. On macOS, open `GPTScratchLab/GPTScratchLab.app`. Keep the bundled runtime, licence notices and manifests together. See the included `DOWNLOADS.md` for launch, checksum and troubleshooting instructions.

Start with the bundled synthetic demonstration text and Tiny CPU settings, or open a UTF-8 corpus you have permission to use. Training and generation remain local. Save a checkpoint to keep model parameters and the character vocabulary; this version does not promise optimizer-state training resumption. The supplied demo is an execution fixture, not evidence of general language quality. The maintained pre-layer-normalization/GELU implementation is distinct from the preserved historical post-normalization tutorial notebooks.

Every native archive must pass an extracted-executable diagnostic covering actual optimizer updates, finite parameters, checkpoint/vocabulary round trips, generation and Tk Train/Generate/save/load/export handlers. The wheel is built from the source distribution and tested after installation outside the checkout. These execution checks do not replace human visible-GUI testing or establish learning quality.

`SHA256SUMS.txt`, three `BUILD-MANIFEST-*.json` files and `PYTHON-DISTRIBUTIONS.json` record exact source provenance, dependency/runtime versions, tested artifact hashes and execution results. Original dependency notices include separate Tcl and Tk terms.

Windows builds are unsigned. macOS builds are ad-hoc signed without Developer ID or notarization; downloaded applications may be blocked by OS publisher checks. Intel Macs and unlisted architectures do not have native packages in this release.
