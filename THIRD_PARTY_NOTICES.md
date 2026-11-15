# Third-party components and learning references

The newly maintained GPT Scratch Lab source is covered by the scoped MIT licence in [LICENSE](LICENSE) and [LICENSE-SCOPE.md](LICENSE-SCOPE.md). Historical notebooks and media keep their source boundaries and are excluded from the maintained Python distribution.

## Runtime and build components

| Component | Role | Upstream source / licence |
| --- | --- | --- |
| CPython | Bundled interpreter | [Python source and notices](https://github.com/python/cpython/blob/main/LICENSE) |
| PyTorch | CPU tensors, model, optimizer and serialization | [BSD-style licence and third-party notices](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| NumPy | Numerical runtime dependency | [NumPy licence](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| Tcl and Tk | Tkinter desktop runtime | [Tcl notice](https://github.com/tcltk/tcl/blob/main/license.terms), [Tk notice](https://github.com/tcltk/tk/blob/main/license.terms) |
| PyInstaller | Native application bundler | [Licence and bundle exception](https://pyinstaller.org/en/stable/license.html) |
| PyInstaller hook contributors | Runtime/dependency collection | [Hook licences](https://github.com/pyinstaller/pyinstaller-hooks-contrib/blob/master/LICENSE) |

Native packages preserve original component notices and record installed versions, actual Tcl/Tk versions and notice provenance. Consult the included notices and each package's build manifest for the exact distribution. These components keep their own terms; the application's MIT licence does not replace them.

## Referenced learning code

The archived study notebook classes follow [Elliot Arledge's freeCodeCamp source](https://github.com/Infatoshi/fcc-intro-to-llms/blob/c7a743092785f4214261dd299633061a06d41434/gpt-v1.ipynb). No explicit upstream licence was found; this file grants no new licence to that material.

[Karpathy's lecture repository](https://github.com/karpathy/ng-video-lecture) and [nanoGPT](https://github.com/karpathy/nanoGPT) are learning references. Their original terms apply to any code taken from them. nanoGPT's [MIT notice](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/LICENSE) identifies Andrej Karpathy's copyright. The maintained package does not bundle their pretrained weights or claim that those repositories are its own original work.

[OpenAI GPT-2](https://github.com/openai/gpt-2) provides the architecture-family reference. The new tiny character decoder is described as **GPT-2-inspired**, with differences documented in [ARCHITECTURE.md](docs/ARCHITECTURE.md). The maintained release does not ship OpenAI's pretrained checkpoints.
