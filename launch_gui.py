import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.environ["ANSYSLMD_LICENSE_FILE"] = "24500@licensing.hkust.edu.cn"

PYTHON = r"D:\miniconda\envs\agents\python.exe"


def main():
    print("=" * 40)
    print("HUDS GUI Launcher")
    print("=" * 40)
    print("[1] Wizard  (PyQt5 step-by-step)")
    print("[2] Tabs    (PyQt5 multi-panel)")
    print("[3] Modern  (PySide6 + Material)")

    choice = input("Choice [1]: ").strip() or "1"

    options = {
        "1": os.path.join(PROJECT_ROOT, "gui_wizard", "main.py"),
        "2": os.path.join(PROJECT_ROOT, "gui_tabs", "main.py"),
        "3": os.path.join(PROJECT_ROOT, "gui_modern", "main.py"),
    }

    target = options.get(choice)
    if not target:
        print("Invalid choice")
        sys.exit(1)

    if not os.path.exists(target):
        print(f"GUI not found: {target}")
        print("Please make sure the GUI module is installed.")
        sys.exit(1)

    subprocess.run([PYTHON, target])


if __name__ == "__main__":
    main()
