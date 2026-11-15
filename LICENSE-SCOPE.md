# Licence scope and source boundaries

The root MIT licence covers the newly maintained `src/gpt_scratch_lab/` package, its original demonstration corpus, `tests/`, desktop packaging/release tooling and documentation added for the 15 November 2026 release. The source distribution is scoped to that maintained application.

It does not assign new rights to the historical `bigram.ipynb`, `gpt-v1 (3).ipynb`, `data/sample.txt`, architecture PNG, demonstration MP4 or their exact preserved copies under `legacy/`. Those remain historical study materials with their existing source and attribution boundaries. The notebook model classes follow Elliot Arledge's freeCodeCamp course source; no explicit licence was found in that upstream repository. See [learning evidence](docs/LEARNING_EVIDENCE.md).

Karpathy's lecture code and nanoGPT are linked learning references. Their original licence terms apply to any code reused from them; this project does not relicense those repositories. The maintained package implements a small, separately structured decoder and does not include their pretrained weights.

Bundled Python, PyTorch, NumPy, Tcl/Tk, PyInstaller and other runtime components keep their own licences. Native packages retain their supplied notices and a dependency/build inventory. Preserve those notices when redistributing a package. A package manifest records the exact versions and origins used for that build.
