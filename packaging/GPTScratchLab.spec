# Keep the standard Torch and _tkinter hooks: Torch source and Tcl/Tk scripts
# are needed at runtime. Build the installed wheel, not an editable checkout.
import json
from pathlib import Path
import sys
from PyInstaller.utils.hooks import collect_data_files

root = Path(SPECPATH).parent
config = json.loads((root / "packaging/release.json").read_text(encoding="utf-8"))
a = Analysis(
    [str(root / "packaging/desktop_entry.py")], pathex=[], binaries=[],
    datas=collect_data_files("gpt_scratch_lab"),
    hiddenimports=["gpt_scratch_lab.gui", "tkinter", "tkinter.ttk", "tkinter.filedialog", "tkinter.messagebox"],
    hookspath=[], hooksconfig={}, runtime_hooks=[], excludes=[], noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name="GPTScratchLab",
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=False, disable_windowed_traceback=False,
    target_arch="arm64" if sys.platform == "darwin" else None,
    codesign_identity=None, entitlements_file=None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="GPTScratchLab")
if sys.platform == "darwin":
    app = BUNDLE(
        coll, name="GPTScratchLab.app", icon=None,
        bundle_identifier="io.github.jyotheshkar.gptscratchlab",
        info_plist={
            "CFBundleDisplayName": "GPT Scratch Lab",
            "CFBundleShortVersionString": config["version"],
            "CFBundleVersion": config["version"],
            "LSMinimumSystemVersion": "15.0",
            "NSHighResolutionCapable": True,
        },
    )
