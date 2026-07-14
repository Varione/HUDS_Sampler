# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import sys

block_cipher = None

SRC_DIR = Path(SPECPATH).resolve()
ENV_PREFIX = Path(sys.executable).resolve().parent
CONDA_BIN = ENV_PREFIX / "Library" / "bin"


def conda_binary(name: str):
    path = CONDA_BIN / name
    return [(str(path), ".")] if path.is_file() else []


def environment_binary(name: str):
    path = ENV_PREFIX / name
    return [(str(path), ".")] if path.is_file() else []


# _ctypes.pyd is collected into _internal. Its libffi dependency must be
# adjacent to it because ctypes is imported before custom runtime hooks run.
extra_binaries = conda_binary("ffi-8.dll") + conda_binary("ffi.dll")
for name in [
    "libcrypto-3-x64.dll", "libssl-3-x64.dll", "zlib.dll", "liblzma.dll",
    "libbz2.dll", "libexpat.dll", "sqlite3.dll", "tcl86t.dll", "tk86t.dll",
]:
    extra_binaries += conda_binary(name)

# Conda's Python runtime is linked against these exact DLLs. Do not rely on
# whatever version happens to be installed in System32 on the target machine.
for name in [
    "vcruntime140.dll", "vcruntime140_1.dll", "vcruntime140_threads.dll",
    "msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll", "msvcp140_codecvt_ids.dll",
]:
    extra_binaries += environment_binary(name)

a = Analysis(
    [str(SRC_DIR / "gui_wizard" / "main.py")],
    pathex=[str(SRC_DIR)],
    binaries=extra_binaries,
    datas=[(str(SRC_DIR / "huds_app"), "huds_app")],
    hiddenimports=[
        "huds_app.core.config", "huds_app.core.storage", "huds_app.core.metrics",
        "huds_app.data.pool", "huds_app.data.schema", "huds_app.data.validation",
        "huds_app.interface.maxwell", "huds_app.interface.maxwell_sweep",
        "huds_app.interface.workflow", "huds_app.model.architecture",
        "huds_app.model.train", "huds_app.sampling.huds",
        "huds_app.utils.aedt_parser", "huds_app.utils.aedt_instances",
        "gui_wizard.worker", "gui_wizard.pages.aedt_page",
        "gui_wizard.pages.config_page", "gui_wizard.pages.monitor_page",
        "gui_wizard.pages.result_page", "PyQt5.QtCore", "PyQt5.QtGui",
        "PyQt5.QtWidgets", "pyqtgraph", "numpy", "pandas", "torch",
        "sklearn", "win32com", "win32com.client",
    ],
    hookspath=[str(SRC_DIR / "hooks")],
    hooksconfig={},
    runtime_hooks=[str(SRC_DIR / "rthooks_fix_path.py")],
    excludes=["PySide6"],
    noarchive=True,
    optimize=-1,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name="HUDS_Wizard",
    debug=False, bootloader_ignore_signals=False, strip=False, upx=True,
    console=False, disable_windowed_traceback=False, argv_emulation=False,
    target_arch=None, codesign_identity=None, entitlements_file=None,
)

coll = COLLECT(
    exe, a.binaries, a.datas, strip=False, upx=True, upx_exclude=[],
    name=str(SRC_DIR / "dist" / "HUDS_Wizard"),
)
