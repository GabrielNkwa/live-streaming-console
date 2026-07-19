# -*- mode: python ; coding: utf-8 -*-
# Build with: pyinstaller dlc-surveillance.spec
#
# onedir (not onefile) deliberately - onefile re-extracts the whole bundle
# to a temp dir on every single launch, which is painfully slow for a
# multi-GB torch/opencv/ultralytics bundle. onedir extracts once at build
# time; distribute the resulting folder (zip it, or wrap it with a proper
# installer - see DESKTOP_BUILD.md).
from PyInstaller.utils.hooks import collect_all

datas = [
    ('templates', 'templates'),
    ('static', 'static'),
    ('yolo11n.pt', '.'),
]
binaries = []
hiddenimports = []

# These packages do a lot of dynamic/conditional importing that PyInstaller's
# static analysis can't see on its own - collect_all pulls in their data
# files, native binaries, and submodules based on each package's own
# PyInstaller hook (where one exists) or heuristics.
for pkg in ('ultralytics', 'torch', 'torchvision', 'cv2'):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

a = Analysis(
    ['desktop.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DLC Surveillance',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # Set to True while debugging a build (keeps a console window showing
    # stdout/stderr); set to False for a real release build so it opens as
    # a clean windowed app with no console.
    console=True,
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
    upx=False,
    upx_exclude=[],
    name='DLC Surveillance',
)
