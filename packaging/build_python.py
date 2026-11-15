"""Build a clean wheel from the sdist and test it outside the checkout."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import tempfile
import zipfile

from release_common import CONFIG, ROOT, check_config, digest, smoke_valid, write_json


def check_member(name: str) -> None:
    path = PurePosixPath(name)
    if (path.is_absolute() or ".." in path.parts or "\\" in name
            or any(part in ("legacy", ".git", ".venv", "__pycache__") for part in path.parts)
            or path.suffix.lower() in (".ipynb", ".mp4", ".pt", ".pth", ".pkl", ".pickle")):
        raise RuntimeError(f"Unexpected historical, unsafe or generated source-package member: {name}")


def build(source_sha: str, output: Path) -> None:
    check_config()
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("Expected a full source commit SHA")
    output.mkdir(parents=True, exist_ok=True)
    expected = CONFIG["distribution"]
    if any((output / expected[kind]).exists() for kind in ("wheel", "sdist")):
        raise RuntimeError("Distribution files already exist; use a fresh output directory")
    # Default build creates the wheel FROM the generated sdist, which proves the
    # sdist contains the maintained package and bundled demo resource.
    subprocess.run([sys.executable, "-m", "build", "--no-isolation", "--outdir", str(output)],
                   cwd=ROOT, check=True)
    wheel, sdist = output / expected["wheel"], output / expected["sdist"]
    if not wheel.is_file() or not sdist.is_file():
        raise RuntimeError("Distribution names do not match the release contract")
    with zipfile.ZipFile(wheel) as archive:
        wheel_members = archive.namelist()
        for name in wheel_members:
            check_member(name)
        if "gpt_scratch_lab/data/demo.txt" not in wheel_members:
            raise RuntimeError("Wheel is missing its offline demo corpus")
    with tarfile.open(sdist) as archive:
        source_members = archive.getmembers()
        for member in source_members:
            check_member(member.name)
            if member.issym() or member.islnk():
                raise RuntimeError("Source distribution must not contain links")
        if not any(member.name.endswith("/src/gpt_scratch_lab/data/demo.txt") for member in source_members):
            raise RuntimeError("Source distribution is missing its demo corpus")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel)], check=True)
    if importlib.metadata.version(expected["name"]) != CONFIG["version"]:
        raise RuntimeError("Installed wheel version differs from the release")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    report = output / "wheel-smoke.json"
    with tempfile.TemporaryDirectory(prefix="gptscratch-wheel-smoke-") as temporary:
        subprocess.run([sys.executable, "-m", "gpt_scratch_lab", "--smoke-test", "--smoke-report", str(report)],
                       cwd=temporary, env=environment, timeout=180, check=True)
        smoke = json.loads(report.read_text(encoding="utf-8"))
        if not smoke_valid(smoke, frozen=False):
            raise RuntimeError(f"Installed wheel failed runtime contract: {smoke}")
        # Resolve imports in a clean subprocess, avoiding the checkout's sys.path.
        location = subprocess.check_output([sys.executable, "-c",
            "import gpt_scratch_lab; print(gpt_scratch_lab.__file__)"], cwd=temporary, env=environment, text=True).strip()
        if Path(location).resolve().is_relative_to((ROOT / "src").resolve()):
            raise RuntimeError("Wheel test imported an editable source checkout")
    manifest = {
        "application": CONFIG["application"], "version": CONFIG["version"], "tag": CONFIG["tag"],
        "repository_id": CONFIG["repository_id"], "source_commit": source_sha,
        "wheel_smoke": smoke, "wheel_built_from_sdist": True,
        "historical_notebooks_included": False,
        "distributions": [{"filename": path.name, "sha256": digest(path), "bytes": path.stat().st_size}
                          for path in (wheel, sdist)],
        "installed_package_path": location,
        "build_dependencies": {name: importlib.metadata.version(name) for name in ("build", "setuptools")},
    }
    write_json(output / "PYTHON-DISTRIBUTIONS.json", manifest)
    print("Verified wheel installed outside the checkout; sdist excludes historical notebooks and media.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "release-assets")
    args = parser.parse_args()
    build(args.source_sha, args.output.resolve())
