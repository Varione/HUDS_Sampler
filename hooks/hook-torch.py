from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_dynamic_libs,
)
import os

# Collect ALL .py source files from torch package as data.
# Critical: torch uses inspect.getsource() at runtime to read its own source code.
# PyInstaller's optimize strips .py in favor of .pyc which breaks this.
# IMPORTANT: all files placed under 'torch/' prefix to avoid shadowing stdlib
# modules (types.py, random.py, signal/, etc.)
torch_pkg = __import__('torch')
torch_dir = os.path.dirname(torch_pkg.__file__)

all_py_files = []
for root, dirs, files in os.walk(torch_dir):
    dirs[:] = [d for d in dirs if not (d.startswith('test') or d == 'lib')]
    for f in files:
        if f.endswith('.py'):
            src = os.path.join(root, f)
            rel = os.path.relpath(root, torch_dir)
            dest = os.path.join('torch', rel)
            all_py_files.append((src, dest))

datas = all_py_files
hiddenimports = collect_submodules('torch')
binaries = collect_dynamic_libs('torch')
