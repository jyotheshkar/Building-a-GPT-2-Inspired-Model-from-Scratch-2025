# Source and maintenance provenance

The repository was created on **20 January 2025**. The `2025` name suffix records that creation year. The maintained desktop/package work is dated **15 November 2026**.

## Inspected baseline

- Repository: `jyotheshkar/Creating-a-Large-Language-Model-from-Scratch-2025`
- Main commit: `604c7fe749c9687b3abf979262de6d4ef395cdeb`
- Git tree: `a74ca8ab4b9af6b905145620ea9266dc53417be4`
- Original demonstration MP4 blob: `14e0b42498448f41fe17b7c80e1818e44eeff898`

The exact two notebooks and default historical text fixture are retained under [legacy/notebooks-before-desktop-2026-11-15](../legacy/notebooks-before-desktop-2026-11-15/), with byte counts, Git blob hashes and SHA-256 values. The architecture image and 25.3 MB original video remain their existing repository blobs. No new recording, historical training result or course certificate was fabricated.

## Historical versus maintained behavior

The earlier notebooks retain the tutorial's post-layernorm/ReLU architecture. Current copies clear stale execution output, generate in inference mode while restoring training state, derive positions from the input tensor's device, and use a prompt made from the selected corpus instead of a hardcoded out-of-vocabulary Dorothy example. Exact originals remain available for comparison.

The new maintained package is a separate tiny GPT-2-inspired pre-layernorm/GELU character decoder with explicit configuration, tokenizer, training, generation and checkpoint boundaries. Original learning-source attribution is documented in [LEARNING_EVIDENCE.md](LEARNING_EVIDENCE.md). New package licensing is scoped in [LICENSE-SCOPE.md](../LICENSE-SCOPE.md).

## Release evidence

Each native archive and Python distribution is built from an exact source commit. Published build manifests record that commit, package/runtime versions and actual checks performed. Checksums identify the uploaded artifacts. Automated execution reports are distinct from the preserved notebook outputs/video and do not establish language-model quality or broad laptop compatibility.
