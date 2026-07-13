from PyQt5.QtWidgets import (
    QMainWindow,
    QTabWidget,
)

from gui_tabs.panels.config_panel import ConfigPanel
from gui_tabs.panels.aedt_panel import AEDTPanel
from gui_tabs.panels.monitor_panel import MonitorPanel
from gui_tabs.panels.result_panel import ResultPanel


class HUDSTabWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HUDS Active Learning")
        self.resize(1000, 700)

        self.tabs = QTabWidget()

        self.config_panel = ConfigPanel()
        self.aedt_panel = AEDTPanel()
        self.monitor_panel = MonitorPanel()
        self.result_panel = ResultPanel()

        self.tabs.addTab(self.config_panel, "Configuration")
        self.tabs.addTab(self.aedt_panel, "AEDT Connection")
        self.tabs.addTab(self.monitor_panel, "Monitor")
        self.tabs.addTab(self.result_panel, "Results")

        self.setCentralWidget(self.tabs)

        self.tabs.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, index):
        if index == 2:
            self._sync_to_monitor()
        if index == 3:
            self._sync_to_result()

    def _sync_to_monitor(self):
        config = self.config_panel.get_config()
        aedt_path, design_name = self.aedt_panel.get_aedt_info()
        if aedt_path:
            config["aedt_project_path"] = aedt_path
        if design_name:
            config["design_name"] = design_name
        self.monitor_panel.set_config(config)
        self.monitor_panel.set_aedt_info(aedt_path, design_name)

    def _sync_to_result(self):
        pass
