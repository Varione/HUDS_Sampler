"""Restore PyTorch source files required by inspect.getsource() after build."""

import os
import shutil
from pathlib import Path

project_root = Path(__file__).resolve().parent
dist_root = project_root / "dist" / "HUDS_Wizard" / "_internal"
dist_torch = dist_root / "torch"
src_torch = Path(__import__("torch").__file__).parent
env_prefix = Path(__import__("sys").executable).resolve().parent

if not dist_torch.is_dir():
    raise FileNotFoundError(f"PyInstaller torch output is missing: {dist_torch}")

copied = 0
for root, dirs, files in os.walk(src_torch):
    dirs[:] = [name for name in dirs if not (name.startswith("test") or name == "lib")]
    for name in files:
        if not name.endswith(".py"):
            continue
        source = Path(root) / name
        destination = dist_torch / Path(root).relative_to(src_torch) / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1

runtime_dlls = [
    "ffi-8.dll", "ffi.dll", "zlib.dll", "vcruntime140.dll",
    "vcruntime140_1.dll", "vcruntime140_threads.dll", "msvcp140.dll",
    "msvcp140_1.dll", "msvcp140_2.dll", "msvcp140_atomic_wait.dll",
    "msvcp140_codecvt_ids.dll",
]
for name in runtime_dlls:
    candidates = [env_prefix / name, env_prefix / "Library" / "bin" / name]
    source = next((path for path in candidates if path.is_file()), None)
    if source:
        shutil.copy2(source, dist_root / name)

# Qt bundles an older VC++ runtime. Keeping it alongside the newer runtime
# required by the current PyTorch wheel causes c10.dll to initialize against a
# mixed CRT set. The newer root copies are backward compatible with Qt.
qt_bin = dist_root / "PyQt5" / "Qt5" / "bin"
for name in ["MSVCP140.dll", "MSVCP140_1.dll", "VCRUNTIME140.dll", "VCRUNTIME140_1.dll"]:
    path = qt_bin / name
    if path.is_file():
        path.unlink()

# Windows supplies the Universal CRT. Conda's private UCRT and API-set stubs
# would load in addition to System32's UCRT and cause the same mixed-runtime
# state before torch is imported.
for path in [dist_root / "ucrtbase.dll", *dist_root.glob("api-ms-win-crt-*.dll")]:
    if path.is_file():
        path.unlink()

print(f"Copied {copied} PyTorch source files to {dist_torch}")
