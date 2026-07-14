import sys
import os

# Fix for PyInstaller: set working directory to exe location
if getattr(sys, 'frozen', False):
    APPLICATION_PATH = os.path.dirname(sys.executable)
else:
    APPLICATION_PATH = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, APPLICATION_PATH)
