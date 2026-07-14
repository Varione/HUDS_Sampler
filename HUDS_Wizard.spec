# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

import os
import sys
# Spec file is in project root - hardcoded for PyInstaller compat
SRC_DIR = r'E:\大型数据库构建\HUDS'

a = Analysis(
    [os.path.join(SRC_DIR, 'gui_wizard', 'main.py')],
    pathex=[SRC_DIR],
    binaries=[],
    datas=[('huds_app', 'huds_app')],
    hiddenimports=[
        'huds_app.core.config',
        'huds_app.core.storage',
        'huds_app.core.metrics',
        'huds_app.data.pool',
        'huds_app.data.schema',
        'huds_app.data.validation',
        'huds_app.interface.maxwell',
        'huds_app.interface.maxwell_sweep',
        'huds_app.interface.workflow',
        'huds_app.model.architecture',
        'huds_app.model.train',
        'huds_app.sampling.huds',
        'huds_app.utils.aedt_parser',
        'huds_app.utils.aedt_instances',
        'gui_wizard.worker',
        'gui_wizard.pages.aedt_page',
        'gui_wizard.pages.config_page',
        'gui_wizard.pages.monitor_page',
        'gui_wizard.pages.result_page',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'pyqtgraph',
        'numpy',
        'pandas',
        'torch',
        'sklearn',
        'win32com',
        'win32com.client',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(SRC_DIR, 'rthooks_fix_path.py')],
    excludes=[
        'PySide6',
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='HUDS_Wizard',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=r'D:\HUDS_Builds\HUDS_Wizard',
)
