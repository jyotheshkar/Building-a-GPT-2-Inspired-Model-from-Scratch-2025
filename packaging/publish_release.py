"""Publish only verified exact-source assets; existing releases are immutable."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

from release_common import (CONFIG, MAX_ASSET_BYTES, ROOT, check_config, digest, expected_asset_names,
                            find_release, smoke_valid, tag_sha, verify_identity)
from release_gate import verify_existing


def validate_assets(source_sha: str, assets: Path) -> list[Path]:
    required = []
    for target, platform in CONFIG["platforms"].items():
        archive = assets / platform["archive"]
        manifest_path = assets / f"BUILD-MANIFEST-{target}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("source_commit") != source_sha or manifest.get("version") != CONFIG["version"]
                or manifest.get("tag") != CONFIG["tag"] or manifest.get("target") != target
                or not smoke_valid(manifest.get("extracted_archive_smoke", {}), frozen=True)
                or manifest["archive"]["sha256"] != digest(archive)
                or manifest["archive"]["bytes"] != archive.stat().st_size
                or set(manifest.get("tcl_tk_runtime", {})) != {"tcl", "tk"}):
            raise RuntimeError(f"Release evidence failed for {target}")
        required.extend((archive, manifest_path))
    path = assets / "PYTHON-DISTRIBUTIONS.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (manifest.get("source_commit") != source_sha or manifest.get("version") != CONFIG["version"]
            or manifest.get("tag") != CONFIG["tag"] or manifest.get("wheel_built_from_sdist") is not True
            or manifest.get("historical_notebooks_included") is not False
            or not smoke_valid(manifest.get("wheel_smoke", {}), frozen=False)):
        raise RuntimeError("Python distribution execution evidence failed")
    if {item["filename"] for item in manifest["distributions"]} != {
            CONFIG["distribution"]["wheel"], CONFIG["distribution"]["sdist"]}:
        raise RuntimeError("Python distribution names differ from release configuration")
    for item in manifest["distributions"]:
        archive = assets / item["filename"]
        if digest(archive) != item["sha256"] or archive.stat().st_size != item["bytes"]:
            raise RuntimeError("Python archive differs from its tested artifact")
        required.append(archive)
    required.append(path)
    if any(not file.is_file() or file.stat().st_size > MAX_ASSET_BYTES for file in required):
        raise RuntimeError("Missing or oversized release artifact")
    checksums = assets / "SHA256SUMS.txt"
    checksums.write_text("".join(f"{digest(file)}  {file.name}\n" for file in sorted(required)),
                         encoding="utf-8", newline="\n")
    required.append(checksums)
    if {file.name for file in required} != expected_asset_names():
        raise RuntimeError("Unexpected final release asset set")
    return required


def publish(source_sha: str, assets: Path) -> None:
    check_config()
    if (not re.fullmatch(r"[0-9a-f]{40}", source_sha)
            or os.environ.get("GITHUB_EVENT_NAME") != "push"
            or os.environ.get("GITHUB_REF") != "refs/heads/main"
            or os.environ.get("GITHUB_SHA") != source_sha):
        raise RuntimeError("Publishing requires the exact main push")
    required = validate_assets(source_sha, assets)
    name = verify_identity()
    existing_tag = tag_sha(name, CONFIG["tag"])
    release = find_release(name, CONFIG["tag"])
    if existing_tag is not None and existing_tag != source_sha:
        raise RuntimeError("Release tag belongs to another source; bump the version, never overwrite")
    if release is not None:
        if existing_tag != source_sha:
            raise RuntimeError("Existing release does not have the exact source tag")
        verify_existing(name, release, source_sha)
        print("Exact-source release already exists; leaving it unchanged.")
        return
    if existing_tag is None:
        subprocess.run(["gh", "api", f"repos/{name}/git/refs", "--method", "POST",
                        "-f", "ref=refs/tags/" + CONFIG["tag"], "-f", "sha=" + source_sha], check=True)
    if tag_sha(name, CONFIG["tag"]) != source_sha:
        raise RuntimeError("Exact-source tag verification failed before draft creation")
    subprocess.run(["gh", "release", "create", CONFIG["tag"], *map(str, required),
                    "--repo", name, "--verify-tag", "--draft",
                    "--title", f"GPT Scratch Lab desktop {CONFIG['version']}",
                    "--notes-file", str(ROOT / "packaging/RELEASE_NOTES.md")], check=True)
    draft = find_release(name, CONFIG["tag"])
    if draft is None or {item["name"] for item in draft["assets"]} != expected_asset_names():
        raise RuntimeError("Draft upload is incomplete; nothing was overwritten or published")
    if any(item.get("state") != "uploaded" or item.get("size", 0) <= 0 for item in draft["assets"]):
        raise RuntimeError("Draft contains an incomplete upload")
    expected = {path.name:path for path in required}
    for asset in draft["assets"]:
        local = expected[asset["name"]]
        if asset["size"] != local.stat().st_size or asset.get("digest") != "sha256:" + digest(local):
            raise RuntimeError("Draft upload digest differs from the verified local asset")
    if tag_sha(name, CONFIG["tag"]) != source_sha:
        raise RuntimeError("Source tag changed; release remains a draft")
    subprocess.run(["gh", "release", "edit", CONFIG["tag"], "--repo", name,
                    "--draft=false", "--latest"], check=True)
    published = find_release(name, CONFIG["tag"])
    if published is None:
        raise RuntimeError("Published release could not be read back")
    verify_existing(name, published, source_sha)
    print(f"Published {CONFIG['tag']} for {source_sha} in repository ID {CONFIG['repository_id']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    publish(args.source_sha, args.assets.resolve())
