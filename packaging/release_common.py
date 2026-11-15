"""Shared release contract; GitHub writes exist only in publish_release.py."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "packaging/release.json").read_text(encoding="utf-8"))
REPOSITORY_ID = 919738589
MAX_ASSET_BYTES = 2 * 1024**3 - 1


def repository() -> str:
    name = os.environ.get("GITHUB_REPOSITORY", "")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", name):
        raise RuntimeError("GITHUB_REPOSITORY must identify the current repository")
    return name


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def api(path: str):
    process = subprocess.run(["gh", "api", path], text=True, capture_output=True)
    if process.returncode:
        if "HTTP 404" in process.stderr:
            return None
        raise RuntimeError(f"GitHub read failed: {process.stderr.strip()}")
    return json.loads(process.stdout)


def verify_identity() -> str:
    name = repository()
    if str(os.environ.get("GITHUB_REPOSITORY_ID")) != str(REPOSITORY_ID):
        raise RuntimeError("This workflow is restricted to the intended repository ID")
    record = api(f"repos/{name}")
    if record is None or record.get("id") != REPOSITORY_ID:
        raise RuntimeError("Repository identity changed")
    return name


def tag_sha(name: str, tag: str):
    reference = api(f"repos/{name}/git/ref/tags/{tag}")
    if reference is None:
        return None
    target = reference["object"]
    for _ in range(5):
        if target["type"] == "commit":
            return target["sha"]
        if target["type"] != "tag":
            break
        target = api(f"repos/{name}/git/tags/{target['sha']}")["object"]
    raise RuntimeError("Release tag does not resolve to a commit")


def find_release(name: str, tag: str):
    # Authenticated release lists include drafts; by-tag endpoints can omit them.
    matches = []
    for page in range(1, 101):
        records = api(f"repos/{name}/releases?per_page=100&page={page}")
        if not isinstance(records, list):
            raise RuntimeError("Could not enumerate existing releases safely")
        matches.extend(item for item in records if item.get("tag_name") == tag)
        if len(records) < 100:
            break
    else:
        raise RuntimeError("Release enumeration limit reached")
    if len(matches) > 1:
        raise RuntimeError("Multiple releases share this tag")
    if not matches:
        return None
    detail = api(f"repos/{name}/releases/{matches[0]['id']}")
    if detail is None or detail.get("tag_name") != tag:
        raise RuntimeError("Release changed during inspection")
    return detail


def check_config() -> None:
    if CONFIG.get("repository_id") != REPOSITORY_ID:
        raise RuntimeError("Release configuration has the wrong repository identity")
    if not re.fullmatch(r"\d+\.\d+\.\d+", CONFIG["version"]):
        raise RuntimeError("Unexpected release version")
    if CONFIG["tag"] != "desktop-v" + CONFIG["version"]:
        raise RuntimeError("Application version/tag mismatch")


def smoke_valid(value: dict, *, frozen: bool) -> bool:
    gui = value.get("gui", {})
    return all((value.get("status") == "passed", value.get("version") == CONFIG["version"],
                value.get("frozen") is frozen, value.get("device") == "cpu",
                value.get("training_steps", 0) >= 2,
                value.get("parameters_changed") is True, value.get("parameters_finite") is True,
                value.get("checkpoint_roundtrip") is True, value.get("tokenizer_roundtrip") is True,
                value.get("generation_length", 0) > 0, value.get("generation_finite") is True,
                value.get("causal_mask_passed") is True,
                gui.get("status") == "passed", gui.get("widgets_constructed") is True,
                gui.get("default_source_loaded") is True, gui.get("initial_idle_state") is True,
                gui.get("gui_training_steps", 0) >= 2, gui.get("gui_generation_length", 0) > 0,
                gui.get("gui_checkpoint_roundtrip") is True, gui.get("gui_text_export") is True))


def expected_asset_names() -> set[str]:
    return ({platform["archive"] for platform in CONFIG["platforms"].values()}
            | {f"BUILD-MANIFEST-{target}.json" for target in CONFIG["platforms"]}
            | {CONFIG["distribution"]["wheel"], CONFIG["distribution"]["sdist"],
               "PYTHON-DISTRIBUTIONS.json", "SHA256SUMS.txt"})
