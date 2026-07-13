import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ["ANSYSLMD_LICENSE_FILE"] = "24500@licensing.hkust.edu.cn"

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    from gui_modern.window import HUDSModernWindow
    window = HUDSModernWindow()
    window.show()
    sys.exit(app.exec())
