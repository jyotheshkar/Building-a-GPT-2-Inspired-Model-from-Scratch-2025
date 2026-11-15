# Verification contracts

Updated 15 November 2026. This document describes the release gates and their practical limits. Actual results belong to the exact source commit's GitHub Actions run and published build manifests.

## Maintained model and package

The source checks exercise character tokenization, configuration bounds, short-corpus validation, finite forward/loss/backward behavior, actual optimizer parameter changes, causal masking, generation/inference state, prompt errors, cancellation, and checkpoint/config/vocabulary round trips. Checkpoint failures must be surfaced without retrying unrestricted pickle loading.

The Python distribution check builds a wheel and source distribution, checks their metadata/content, installs the built package away from the checkout and exercises the installed package. Distribution inputs are scoped to the newly maintained application rather than the historical notebooks/video.

## Extracted native application

Every supported native target is built on its own OS/CPU runner. The final archive is extracted into a separate directory, then the extracted executable runs from an unrelated working directory. Its bounded diagnostic must perform real CPU training updates, verify finite/changed parameters and causal masking, save/reopen a checkpoint, preserve its vocabulary, generate a nonempty sequence and exercise the Tk interface through a withdrawn window and event-loop updates. macOS signature integrity is checked after extraction.

The diagnostic produces machine-readable evidence in platform manifests. Passing does not establish a fluent model, a benchmark score, a completed tutorial, or a human test of every visible desktop state on every supported OS version.

## Historical notebooks

`scripts/notebook_smoke.py` extracts the actual notebook model definitions and checks forward/backward, generation, causality and batch boundaries on a deliberately tiny CPU configuration. The original 2025 notebook training settings are not a release smoke workload. Historical outputs are cleared in maintained copies and preserved separately with the original bytes.

## Release publication

Publication requires the source/package and all native gates to pass. The release tag and manifests must identify the exact tested source. Existing published version assets are retained; new release work requires an explicit version change. Checksums and dependency/Tcl/Tk notices accompany the final artifacts.

Inspect the [desktop workflow](../.github/workflows/desktop-release.yml), the [Releases page](https://github.com/jyotheshkar/Creating-a-Large-Language-Model-from-Scratch-2025/releases), and each release's manifests for actual conclusions. Writing a test or adding a workflow does not by itself establish a pass.
