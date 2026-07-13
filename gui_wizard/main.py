import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.environ["ANSYSLMD_LICENSE_FILE"] = "24500@licensing.hkust.edu.cn"

from PyQt5.QtWidgets import QApplication
from gui_wizard.wizard import HUDSWizard

if __name__ == "__main__":
    app = QApplication(sys.argv)
    wizard = HUDSWizard()
    wizard.show()
    sys.exit(app.exec_())
