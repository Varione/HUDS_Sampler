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

        aedt_page = AEDTPage()
        self._aedt_id = self.addPage(aedt_page)

        cfg_page = ConfigPage()
        self._cfg_id = self.addPage(cfg_page)

        mon_page = MonitorPage()
        self._mon_id = self.addPage(mon_page)

        res_page = ResultPage()
        self._res_id = self.addPage(res_page)

        aedt_page.set_next_id(self._cfg_id)
        cfg_page.set_next_id(self._mon_id)
        mon_page.set_next_id(self._res_id)

    def nextId(self):
        page = self.currentPage()
        if hasattr(page, "nextId"):
            return page.nextId()
        return -1
