# Learning sources and engineering evidence

Updated 15 November 2026. This page separates historical study artifacts, attributable source lineage, requested learning references and the maintained application.

## Historical notebook lineage

The original GPT notebook's five model classes structurally match the historical Elliot Arledge/freeCodeCamp source: `Head`, `MultiHeadAttention`, `FeedFoward`, `Block` and `GPTLanguageModel`. The comparison parses both originals with Python's AST and compares class structures without source-location attributes. The claim concerns the archived pre-release source, rather than the maintained inference fixes.

- [Elliot Arledge / freeCodeCamp video](https://www.youtube.com/watch?v=UU1WVnMk4E8)
- [Official course article](https://www.freecodecamp.org/news/how-to-build-a-large-language-model-from-scratch-using-python/)
- [Historical source notebook](https://github.com/Infatoshi/fcc-intro-to-llms/blob/c7a743092785f4214261dd299633061a06d41434/gpt-v1.ipynb), Git blob `83bb544dba0f867e05bcb1500b15c12496a3f40d`
- [Exact retained notebook copies and hashes](../legacy/notebooks-before-desktop-2026-11-15/manifest.json)

The course article identifies Elliot as the instructor and credits Karpathy as inspiration. No explicit licence was found in the upstream notebook repository. This project preserves that attribution and does not apply the maintained package's MIT licence to those historical files.

## Andrej Karpathy references

These are the requested learning resources for understanding and extending the project:

| Reference | Learning focus |
| --- | --- |
| [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) | Attention, decoder blocks, training and autoregressive sampling |
| [Exact lecture repository](https://github.com/karpathy/ng-video-lecture) | The lecture's implementation sequence and code |
| [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) | Progression from backpropagation/bigrams to GPT and tokenizers |
| [nanoGPT](https://github.com/karpathy/nanoGPT) | Further reference for GPT training and a reduced CPU recipe |
| [OpenAI GPT-2 model source](https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/model.py) | The numbered architecture family behind the new release's qualified name |

Links and code similarities establish reference/source relationships. They do not establish video-watch history, completion of a course, a certificate, independent invention of the tutorial architecture, or pretrained-model quality.

## What the retained work demonstrates

| Retained evidence | What it supports |
| --- | --- |
| [Bigram notebook](../bigram.ipynb) | Character IDs, next-token logits, cross-entropy and sampling baseline |
| [GPT study notebook](../gpt-v1%20%283%29.ipynb) | Causal attention, multi-head projection, residual blocks and training loop study |
| [Historical architecture image](../gpt%20architecture.png) | The earlier post-normalization architecture explanation |
| [Original demonstration video](../Gpt%20model%20test%20prediction.mp4) | A historical recorded demonstration, not evidence of the new release's test results |
| [Preserved source manifest](../legacy/notebooks-before-desktop-2026-11-15/manifest.json) | Exact content retained before the desktop changes |
| [Maintained package](../src/gpt_scratch_lab/) | Explicit configuration/tokenizer/model boundaries and local train/generate/checkpoint interfaces |
| [Tests and release workflow](VERIFICATION.md) | The exact behaviors exercised by source, installed-package and extracted-native checks |

## Added engineering work

The maintained package separates CPU training, generation, tokenization and checkpoint persistence from the desktop/CLI surfaces. It adds bounded defaults, clear prompt validation, inference mode, saved vocabulary/configuration, background work with cancellation, distribution tooling and inspectable build manifests. The new pre-normalization/GELU decoder is described separately from the old post-normalization/ReLU notebook.

Claims about learning quality require a suitable corpus, controlled evaluation and recorded results. The synthetic demonstration and startup/optimizer tests only establish the execution contracts listed in [verification](VERIFICATION.md).
