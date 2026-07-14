from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QListWidget,
    QStackedWidget,
    QVBoxLayout,
    QLabel,
)

from gui_modern.style import DARK_STYLESHEET
from gui_modern.pages.config_page import ConfigPage
from gui_modern.pages.aedt_page import AEDTPage
from gui_modern.pages.monitor_page import MonitorPage
from gui_modern.pages.result_page import ResultPage


class HUDSModernWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HUDS Active Learning")
        self.resize(1100, 750)
        self.setStyleSheet(DARK_STYLESHEET)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        sidebar_container = QWidget()
        sidebar_container.setMaximumWidth(220)
        sidebar_container.setMinimumWidth(180)
        sidebar_layout = QVBoxLayout(sidebar_container)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)

        sidebar_title = QWidget()
        title_layout = QHBoxLayout(sidebar_title)
        title_label = SidebarTitle("HUDS")
        title_layout.addWidget(title_label)
        sidebar_layout.addWidget(sidebar_title)

        self.stack = QStackedWidget()
        self.config_page = ConfigPage()
        self.aedt_page = AEDTPage()
        self.monitor_page = MonitorPage()
        self.result_page = ResultPage()

        self.stack.addWidget(self.aedt_page)
        self.stack.addWidget(self.config_page)
        self.stack.addWidget(self.monitor_page)
        self.stack.addWidget(self.result_page)

        self.sidebar = QListWidget()
        items = ["AEDT Connection", "Configuration", "Monitor", "Results"]
        for item_text in items:
            self.sidebar.addItem(item_text)
        self.sidebar.currentRowChanged.connect(self._on_nav_changed)
        self.sidebar.setCurrentRow(0)
        sidebar_layout.addWidget(self.sidebar)

        separator = SeparatorWidget()

        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(sidebar_container)
        content_layout.addWidget(separator)
        content_layout.addWidget(self.stack, stretch=1)

        main_layout.addLayout(content_layout)

    def _on_nav_changed(self, row):
        self.stack.setCurrentIndex(row)
        if row == 1:
            detected_vars = self.aedt_page.get_detected_variables()
            if detected_vars:
                self.config_page.set_detected_variables(detected_vars)
            
            detected_outputs = self.aedt_page.get_detected_outputs()
            if detected_outputs:
                self.setProperty("detected_outputs", detected_outputs)
                self.config_page._auto_fill_outputs()


class SidebarTitle(QWidget):
    def __init__(self, text):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 20, 16, 12)
        label = QLabel(text)
        label.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #569cd6; "
            "background-color: #252526;"
        )
        layout.addWidget(label)


class SeparatorWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(1)
        self.setStyleSheet("background-color: #3c3c3c;")
