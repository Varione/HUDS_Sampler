# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['gui_tabs/main.py'],
    pathex=[],
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
        'gui_tabs.worker',
        'gui_tabs.panels.aedt_panel',
        'gui_tabs.panels.config_panel',
        'gui_tabs.panels.monitor_panel',
        'gui_tabs.panels.result_panel',
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
    runtime_hooks=[],
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
    name='HUDS_Tabs',
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
    name='D:\\HUDS_Builds\\HUDS_Tabs',
)
