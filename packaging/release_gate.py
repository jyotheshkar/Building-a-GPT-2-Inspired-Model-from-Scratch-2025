"""Build a new desktop version; verify and skip an already published version."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from release_common import (CONFIG, check_config, digest, expected_asset_names, find_release,
                            smoke_valid, tag_sha, verify_identity)


def verify_existing(name: str, release: dict, tagged_sha: str) -> None:
    if release.get("draft") or release.get("prerelease") or not tagged_sha:
        raise RuntimeError("Existing release/tag is incomplete; inspect it before retrying")
    assets = {item["name"]:item for item in release["assets"]}
    if set(assets) != expected_asset_names():
        raise RuntimeError("Existing release has an unexpected or incomplete asset set")
    if any(item.get("state") != "uploaded" or item.get("size", 0) <= 0 for item in assets.values()):
        raise RuntimeError("Existing release has incomplete assets")
    with tempfile.TemporaryDirectory(prefix="gptscratch-release-check-") as temporary:
        temporary = Path(temporary)
        subprocess.run(["gh", "release", "download", CONFIG["tag"], "--repo", name,
                        "--pattern", "BUILD-MANIFEST-*.json", "--pattern", "PYTHON-DISTRIBUTIONS.json",
                        "--dir", str(temporary)], check=True)
        for target, platform in CONFIG["platforms"].items():
            path = temporary / f"BUILD-MANIFEST-{target}.json"
            if assets[path.name].get("digest") != "sha256:" + digest(path):
                raise RuntimeError("Published manifest digest mismatch")
            manifest = json.loads(path.read_text(encoding="utf-8"))
            archive = assets[platform["archive"]]
            if (manifest.get("source_commit") != tagged_sha or manifest.get("version") != CONFIG["version"]
                    or manifest.get("tag") != CONFIG["tag"] or manifest.get("target") != target
                    or not smoke_valid(manifest.get("extracted_archive_smoke", {}), frozen=True)
                    or archive.get("digest") != "sha256:" + manifest["archive"]["sha256"]
                    or archive["size"] != manifest["archive"]["bytes"]):
                raise RuntimeError("Published native evidence differs from its tag/assets")
        path = temporary / "PYTHON-DISTRIBUTIONS.json"
        if assets[path.name].get("digest") != "sha256:" + digest(path):
            raise RuntimeError("Published distribution manifest digest mismatch")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (manifest.get("source_commit") != tagged_sha or manifest.get("version") != CONFIG["version"]
                or manifest.get("tag") != CONFIG["tag"]
                or not smoke_valid(manifest.get("wheel_smoke", {}), frozen=False)):
            raise RuntimeError("Published Python distribution evidence differs from its tag")
        if {item["filename"] for item in manifest["distributions"]} != {
                CONFIG["distribution"]["wheel"], CONFIG["distribution"]["sdist"]}:
            raise RuntimeError("Published Python distribution names differ")
        for item in manifest["distributions"]:
            asset = assets[item["filename"]]
            if asset.get("digest") != "sha256:" + item["sha256"] or asset["size"] != item["bytes"]:
                raise RuntimeError("Published Python archive digest mismatch")


def should_build(source_sha: str) -> bool:
    check_config()
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("Expected an exact source commit")
    if os.environ.get("GITHUB_EVENT_NAME") == "pull_request":
        return True  # Pull requests never write releases or need a privileged token.
    if (os.environ.get("GITHUB_EVENT_NAME") != "push" or os.environ.get("GITHUB_REF") != "refs/heads/main"
            or os.environ.get("GITHUB_SHA") != source_sha):
        raise RuntimeError("Unexpected release trigger")
    name = verify_identity()
    tagged_sha = tag_sha(name, CONFIG["tag"])
    release = find_release(name, CONFIG["tag"])
    if release is None:
        if tagged_sha is not None:
            raise RuntimeError("A tag already exists without a complete release; inspect it manually")
        return True
    # Later main commits with the SAME release version verify existing immutable
    # evidence and skip rebuilding. A new release requires a version/tag bump.
    verify_existing(name, release, tagged_sha)
    print(f"{CONFIG['tag']} is already published for {tagged_sha}; no rebuild or overwrite.")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    result = should_build(args.source_sha)
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write("build=" + str(result).lower() + "\n")
    print("Build native artifacts: " + str(result).lower())
