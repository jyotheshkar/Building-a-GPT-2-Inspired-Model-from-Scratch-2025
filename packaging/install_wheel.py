"""Install this workflow's verified wheel for native analysis."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from release_common import CONFIG, digest


def install(assets: Path, source_sha: str) -> None:
    manifest = json.loads((assets / "PYTHON-DISTRIBUTIONS.json").read_text(encoding="utf-8"))
    if manifest.get("source_commit") != source_sha or manifest.get("version") != CONFIG["version"]:
        raise RuntimeError("Downloaded wheel manifest has the wrong source/version")
    name = CONFIG["distribution"]["wheel"]
    item = next(item for item in manifest["distributions"] if item["filename"] == name)
    wheel = assets / name
    if wheel.stat().st_size != item["bytes"] or digest(wheel) != item["sha256"]:
        raise RuntimeError("Downloaded wheel differs from its tested artifact")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel)], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    arguments = parser.parse_args()
    install(arguments.assets.resolve(), arguments.source_sha)
