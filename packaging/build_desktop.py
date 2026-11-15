"""Build and verify an extracted native desktop bundle on its target OS."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.request import urlopen
import zipfile

from release_common import CONFIG, MAX_ASSET_BYTES, ROOT, check_config, digest, smoke_valid, write_json


def notice_candidate(path: Path) -> bool:
    return any(word in str(path).lower() for word in
               ("license", "licence", "copying", "notice", "copyright", "patents", "lgpl"))


def safe_notice_path(path: str) -> Path:
    return Path(*[part for part in Path(path).parts
                  if part not in ("..", ".", "/", "\\") and not part.endswith(":")])


def collect_notices(destination: Path) -> tuple[list[dict], list[str]]:
    inventory, notices, covered = [], [], set()
    for dist in sorted(importlib.metadata.distributions(), key=lambda d:d.metadata.get("Name", "").lower()):
        name, version = dist.metadata.get("Name", "unknown"), dist.version
        inventory.append({"name": name, "version": version,
                          "home_page": dist.metadata.get("Home-page"),
                          "project_urls": dist.metadata.get_all("Project-URL", []),
                          "license_expression": dist.metadata.get("License-Expression"),
                          "license_metadata": dist.metadata.get("License")})
        component = re.sub(r"[^A-Za-z0-9_.-]", "_", name + "-" + version)
        for file in dist.files or []:
            if not notice_candidate(Path(str(file))):
                continue
            source = Path(dist.locate_file(file))
            if (not source.is_file() or source.stat().st_size > 10 * 1024 * 1024
                    or source.suffix.lower() in (".pyc", ".pyd", ".dll", ".so", ".dylib", ".a")):
                continue
            target = destination / component / safe_notice_path(str(file))
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            notices.append(target.relative_to(destination).as_posix())
            covered.add(name.lower().replace("_", "-"))
    for name in ("torch", "numpy", "pyinstaller", "pyinstaller-hooks-contrib"):
        if name not in covered:
            raise RuntimeError(f"Missing original notices for {name}")
    target = destination / "CPython/LICENSE.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    candidate = next((Path(sys.base_prefix)/name for name in ("LICENSE.txt", "LICENSE", "License.txt")
                      if (Path(sys.base_prefix)/name).is_file()), None)
    if candidate:
        shutil.copyfile(candidate, target)
    else:
        url = f"https://raw.githubusercontent.com/python/cpython/v{platform.python_version()}/LICENSE"
        with urlopen(url, timeout=60) as response:
            target.write_bytes(response.read(1024 * 1024))
    notices.append("CPython/LICENSE.txt")
    return inventory, sorted(notices)


def collect_tcl_tk_notices(destination: Path) -> tuple[dict, list[str]]:
    """Query actual runtimes and preserve each component's own original notice."""
    import tkinter

    window = tkinter.Tk()
    try:
        window.withdraw()
        window.update_idletasks()
        components = {
            "tcl": {"version": str(window.tk.call("info", "patchlevel")),
                    "script_directory": str(window.tk.call("info", "library"))},
            "tk": {"version": str(window.tk.call("package", "present", "Tk")),
                   "script_directory": str(window.tk.call("set", "tk_library"))},
        }
    finally:
        window.destroy()
    notices = []
    for name, component in components.items():
        version = component["version"]
        if not re.fullmatch(r"\d+\.\d+\.\d+", version):
            raise RuntimeError(f"Stable exact Tcl/Tk runtime required: {name} {version}")
        if tuple(map(int, version.split("."))) < (8, 6, 0):
            raise RuntimeError("A supported Tcl/Tk 8.6+ runtime is required")
        source = Path(component["script_directory"]) / "license.terms"
        target = destination / f"{name}-{version}" / "license.terms"
        target.parent.mkdir(parents=True, exist_ok=True)
        exact_tag = "core-" + version.replace(".", "-")
        official_url = f"https://raw.githubusercontent.com/tcltk/{name}/{exact_tag}/license.terms"
        if source.is_file():
            shutil.copyfile(source, target)
            origin = str(source)
        else:
            # Script distributions sometimes omit this file; never use main or
            # assume Tcl and Tk have the same version or identical notice.
            with urlopen(official_url, timeout=60) as response:
                target.write_bytes(response.read(1024 * 1024))
            origin = official_url
        if not target.read_bytes().strip():
            raise RuntimeError(f"Empty original notice for {name}")
        component.update({"notice_origin": origin, "official_notice_url": official_url,
                          "notice_sha256": digest(target),
                          "source_url": f"https://github.com/tcltk/{name}/tree/{exact_tag}",
                          "macos_system_framework": component["script_directory"].startswith("/System/Library/Frameworks/")})
        notices.append(target.relative_to(destination).as_posix())
    print("Preserved Tcl/Tk notices: " + json.dumps(components, sort_keys=True))
    return components, notices


def run_smoke(executable: Path, cwd: Path, report: Path) -> dict:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    subprocess.run([str(executable.resolve()), "--smoke-test", "--smoke-report", str(report.resolve())],
                   cwd=cwd, env=environment, timeout=180, check=True)
    value = json.loads(report.read_text(encoding="utf-8"))
    if not smoke_valid(value, frozen=True):
        raise RuntimeError(f"Extracted application failed execution contract: {value}")
    return value


def make_archive(stage: Path, archive: Path, target: str) -> None:
    if target == "macos-arm64":
        subprocess.run(["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent",
                        str(stage / "GPTScratchLab"), str(archive)], check=True)
    elif target == "linux-x64":
        with tarfile.open(archive, "w:gz", dereference=False) as stream:
            stream.add(stage / "GPTScratchLab", arcname="GPTScratchLab")
    else:
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as stream:
            for path in sorted((stage / "GPTScratchLab").rglob("*")):
                if path.is_file():
                    stream.write(path, path.relative_to(stage))


def extract_and_verify(archive: Path, target: str, config: dict, output: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="gptscratch-native-") as temporary:
        temporary = Path(temporary)
        extracted = temporary / "extracted"
        extracted.mkdir()
        if target == "macos-arm64":
            subprocess.run(["ditto", "-x", "-k", str(archive), str(extracted)], check=True)
            subprocess.run(["codesign", "--verify", "--deep", "--strict",
                            str(extracted / "GPTScratchLab/GPTScratchLab.app")], check=True)
        elif target == "linux-x64":
            # Only this builder's newly generated archive is extracted here.
            with tarfile.open(archive) as stream:
                stream.extractall(extracted)
        else:
            with zipfile.ZipFile(archive) as stream:
                stream.extractall(extracted)
        unrelated = temporary / "unrelated-working-directory"
        unrelated.mkdir()
        return run_smoke(extracted / config["executable_path"], unrelated, output / f"smoke-{target}.json")


def build(target: str, source_sha: str, output: Path) -> None:
    check_config()
    platform_config = CONFIG["platforms"][target]
    if (platform.system() != platform_config["system"]
            or platform.machine().lower() != platform_config["machine"].lower()):
        raise RuntimeError("Build each package on its native target runner")
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("Expected an exact 40-character source commit")
    if importlib.metadata.version(CONFIG["distribution"]["name"]) != CONFIG["version"]:
        raise RuntimeError("Installed application and release versions differ")
    output.mkdir(parents=True, exist_ok=True)
    work = ROOT / "build/desktop" / target
    if work.exists():
        raise RuntimeError(f"Build directory exists: {work}; use a fresh checkout")
    dist, stage = work / "dist", work / "stage"
    work.mkdir(parents=True)
    subprocess.run([sys.executable, "-m", "PyInstaller", str(ROOT / "packaging/GPTScratchLab.spec"),
                    "--noconfirm", "--clean", "--distpath", str(dist),
                    "--workpath", str(work / "pyinstaller")], cwd=ROOT, check=True)
    payload = stage / "GPTScratchLab"
    if target == "macos-arm64":
        payload.mkdir(parents=True)
        shutil.copytree(dist / "GPTScratchLab.app", payload / "GPTScratchLab.app", symlinks=True)
    else:
        shutil.copytree(dist / "GPTScratchLab", payload, symlinks=True)
    inventory, notices = collect_notices(payload / "licenses")
    tcl_tk, gui_notices = collect_tcl_tk_notices(payload / "licenses")
    notices = sorted(set(notices + gui_notices))
    notice_text = (
        "GPT Scratch Lab - third-party components\n\n"
        "Dependency components retain their original licences. Read the original notices in licenses/.\n"
        "The root LICENSE describes the scope of the maintained application's own licence.\n"
        "CPython, CPU PyTorch, NumPy, Tcl/Tk, PyInstaller and contrib-hook notices are retained.\n"
        "The default PyInstaller Tcl/Tk hook preserves bundled script headers. Some macOS system frameworks\n"
        "remain supplied by macOS; the build manifest records the actual queried runtimes.\n"
        "Installed build distributions below are an environment inventory, not a claim every module is bundled.\n\n"
        + "\n".join(f"- {item['name']} {item['version']}" for item in inventory)
        + "\n\nOriginal notice paths:\n" + "\n".join(notices) + "\n"
    )
    (payload / "THIRD-PARTY-LICENSES.txt").write_text(notice_text, encoding="utf-8", newline="\n")
    for source, filename in ((ROOT / "LICENSE", "LICENSE"),
                             (ROOT / "LICENSE-SCOPE.md", "LICENSE-SCOPE.md"),
                             (ROOT / "docs/DOWNLOADS.md", "DOWNLOADS.md"),
                             (ROOT / "packaging/BUILDING.md", "BUILDING.md"),
                             (ROOT / "packaging/RELEASE_NOTES.md", "RELEASE_NOTES.md")):
        shutil.copyfile(source, payload / filename)
    name = os.environ.get("GITHUB_REPOSITORY", "jyotheshkar/Creating-a-Large-Language-Model-from-Scratch-2025")
    manifest = {"application": CONFIG["application"], "version": CONFIG["version"], "tag": CONFIG["tag"],
                "repository_id": CONFIG["repository_id"], "source_repository": "https://github.com/" + name,
                "source_commit": source_sha, "target": target,
                "python": platform.python_version(), "build_platform": platform.platform(),
                "machine": platform.machine(), "dependencies": inventory,
                "preserved_notice_files": notices, "tcl_tk_runtime": tcl_tk,
                "signing": "ad-hoc; no Developer ID or notarization" if target == "macos-arm64" else "unsigned",
                "model_quality": "not evaluated; no pretrained language-model weights supplied"}
    with tempfile.TemporaryDirectory(prefix="gptscratch-stage-smoke-") as temporary:
        manifest["pre_archive_smoke"] = run_smoke(stage / platform_config["executable_path"],
                                                 Path(temporary), work / "pre-archive-smoke.json")
    write_json(payload / "BUILD-MANIFEST.json", manifest)
    archive = output / platform_config["archive"]
    make_archive(stage, archive, target)
    if archive.stat().st_size > MAX_ASSET_BYTES:
        raise RuntimeError("Native archive exceeds GitHub's per-asset limit")
    manifest["extracted_archive_smoke"] = extract_and_verify(archive, target, platform_config, output)
    manifest["archive"] = {"filename": archive.name, "sha256": digest(archive), "bytes": archive.stat().st_size}
    write_json(output / f"BUILD-MANIFEST-{target}.json", manifest)
    print(f"Verified native package: {archive.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=CONFIG["platforms"], required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "release-assets")
    args = parser.parse_args()
    build(args.target, args.source_sha, args.output.resolve())
