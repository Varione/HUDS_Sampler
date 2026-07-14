import sys
import os

# AEDT and PyTorch both bundle Intel OpenMP. PyTorch is loaded first by
# gui_wizard.main, and this setting prevents a later AEDT COM load from
# rejecting the already-loaded runtime.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# Fix for PyInstaller: use _MEIPASS (the _internal temp directory) as application root
if getattr(sys, 'frozen', False):
    APPLICATION_PATH = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
else:
    APPLICATION_PATH = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, APPLICATION_PATH)

# DLL search path fix - MUST run before any torch import
# Critical: hold references to prevent garbage collection of add_dll_directory handles
_dll_handles = []

def fix_dll_paths():
    global _dll_handles
    if sys.platform != 'win32':
        return

    # torch/lib contains c10.dll, torch_cpu.dll, CUDA DLLs etc.
    torch_lib = os.path.join(APPLICATION_PATH, 'torch', 'lib')
    if os.path.isdir(torch_lib):
        try:
            _dll_handles.append(os.add_dll_directory(torch_lib))
        except (OSError, AttributeError):
            pass

    # Root _internal contains Python DLLs, libcrypto, etc.
    try:
        _dll_handles.append(os.add_dll_directory(APPLICATION_PATH))
    except (OSError, AttributeError):
        pass

    dlls_dir = os.path.join(APPLICATION_PATH, 'DLLs')
    if os.path.isdir(dlls_dir):
        try:
            _dll_handles.append(os.add_dll_directory(dlls_dir))
        except (OSError, AttributeError):
            pass

# Must run immediately - before any import that might trigger torch loading
fix_dll_paths()

# Also set PATH as fallback - some modules rely on PATH instead of add_dll_directory
torch_lib = os.path.join(APPLICATION_PATH, 'torch', 'lib')
if os.path.isdir(torch_lib):
    if 'PATH' in os.environ:
        os.environ['PATH'] = torch_lib + ';' + os.environ['PATH']
    else:
        os.environ['PATH'] = torch_lib

dlls_dir = os.path.join(APPLICATION_PATH, 'DLLs')
if os.path.isdir(dlls_dir):
    os.environ['PATH'] = dlls_dir + ';' + os.environ.get('PATH', '')

# Add conda Library/bin to PATH (for zlib, libcrypto, etc.)
conda_bin = os.path.join(APPLICATION_PATH, 'Library', 'bin')
if os.path.isdir(conda_bin):
    if 'PATH' in os.environ:
        os.environ['PATH'] = conda_bin + ';' + os.environ['PATH']
