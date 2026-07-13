from PyQt5.QtWidgets import QWizard

from gui_wizard.pages.config_page import ConfigPage
from gui_wizard.pages.aedt_page import AEDTPage
from gui_wizard.pages.monitor_page import MonitorPage
from gui_wizard.pages.result_page import ResultPage


class HUDSWizard(QWizard):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HUDS 主动学习向导")
        self.setWizardStyle(QWizard.ModernStyle)

        self.addPage(ConfigPage())
        self.addPage(AEDTPage())
        self.addPage(MonitorPage())
        self.addPage(ResultPage())

    def nextId(self):
        current = self.currentPageId()
        page = self.page(current)
        if hasattr(page, "nextId"):
            return page.nextId()
        return super().nextId()
