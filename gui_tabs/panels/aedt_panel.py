from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
)


class AEDTSignals(QObject):
    connected = pyqtSignal()


class AEDTPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._signals = AEDTSignals()
        self.oApp = None
        self.oDesktop = None
        self.oProject = None
        self._setup_ui()

    @property
    def connected_signal(self):
        return self._signals.connected

    def _setup_ui(self):
        layout = QGridLayout(self)
        layout.setSpacing(6)
        row = 0

        # Status
        self.status_label = QLabel("Not connected")
        self.status_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.status_label, row, 0, 1, 3)
        row += 1

        # Connect button
        self.connect_btn = QPushButton("Connect AEDT")
        self.connect_btn.clicked.connect(self._on_connect)
        layout.addWidget(self.connect_btn, row, 0, 1, 3)
        row += 1

        # Instance selector
        layout.addWidget(QLabel("Instance:"), row, 0)
        self.instance_combo = QComboBox()
        self.instance_combo.setVisible(False)
        self.instance_combo.currentIndexChanged.connect(self._on_instance_changed)
        layout.addWidget(self.instance_combo, row, 1, 1, 2)
        row += 1

        # Project selector
        layout.addWidget(QLabel("Project:"), row, 0)
        self.project_combo = QComboBox()
        self.project_combo.setEnabled(False)
        self.project_combo.currentIndexChanged.connect(self._on_project_changed)
        layout.addWidget(self.project_combo, row, 1, 1, 2)
        row += 1

        # Design selector
        layout.addWidget(QLabel("Design:"), row, 0)
        self.design_combo = QComboBox()
        self.design_combo.setEnabled(False)
        self.design_combo.currentIndexChanged.connect(self._on_design_changed)
        layout.addWidget(self.design_combo, row, 1, 1, 2)

        self._detected_vars = []

    def _on_connect(self):
        try:
            from huds_app.utils.aedt_instances import enumerate_aedt_instances
            from huds_app.interface.maxwell_sweep import ensure_aedt_running

            instances = enumerate_aedt_instances()

            if not instances:
                ensure_aedt_running()
                instances = enumerate_aedt_instances()

            if not instances:
                self.status_label.setText("Failed to connect AEDT")
                return

            if len(instances) == 1:
                self.oApp = instances[0]['oApp']
                self.oDesktop = instances[0]['oDesktop']
                self.instance_combo.setVisible(False)
            else:
                self.instance_combo.clear()
                for inst in instances:
                    self.instance_combo.addItem(inst['label'])
                self.instance_combo.setVisible(True)
                self._on_instance_changed(0)
                return

            version = self.oDesktop.GetVersion()
            self.status_label.setText(f"Connected (AEDT {version})")
            self.status_label.setStyleSheet("color: green;")
            self.project_combo.setEnabled(True)
            self.design_combo.setEnabled(True)

            projects = self.oDesktop.GetProjects()
            self.project_combo.clear()
            for i in range(projects.Count):
                name = projects(i).GetName()
                try:
                    path = projects(i).GetPath()
                except Exception:
                    path = "Unknown"
                self.project_combo.addItem(f"{name} ({path})")

            self._signals.connected.emit()
        except Exception as e:
            self.status_label.setText(f"Connection error: {e}")
            self.status_label.setStyleSheet("color: red;")

    def _on_instance_changed(self, index):
        from huds_app.utils.aedt_instances import enumerate_aedt_instances
        instances = enumerate_aedt_instances()
        if 0 <= index < len(instances):
            self.oApp = instances[index]['oApp']
            self.oDesktop = instances[index]['oDesktop']
            version = instances[index]['version']
            self.status_label.setText(f"Connected (AEDT {version})")
            self.status_label.setStyleSheet("color: green;")
            self.project_combo.setEnabled(True)
            self.design_combo.setEnabled(True)

            projects = self.oDesktop.GetProjects()
            self.project_combo.clear()
            for i in range(projects.Count):
                name = projects(i).GetName()
                try:
                    path = projects(i).GetPath()
                except Exception:
                    path = "Unknown"
                self.project_combo.addItem(f"{name} ({path})")

            self._signals.connected.emit()

    def _on_project_changed(self, index):
        if index < 0 or not self.oDesktop:
            return

        projects = self.oDesktop.GetProjects()
        if index < projects.Count:
            self.oProject = projects(index)
        else:
            return

        designs = self.oProject.GetDesigns()
        self.design_combo.clear()
        for i in range(designs.Count):
            self.design_combo.addItem(designs(i).GetName())

    def _on_design_changed(self, index):
        if index < 0 or not self.oProject:
            return
        design_name = self.design_combo.currentText()
        try:
            oDesign = self.oProject.SetActiveDesign(design_name)
        except Exception:
            return

        detected_vars = []
        try:
            aedt_path = self.oProject.GetPath()
            from huds_app.utils.aedt_parser import parse_aedt_variables
            detected_vars = parse_aedt_variables(aedt_path, design_name)
        except Exception as e:
            print(f"[AEDT] Variable detection failed: {e}")

        self._detected_vars = detected_vars
        print(f"[AEDT] Detected {len(detected_vars)} variables:")
        for v in detected_vars:
            print(f"  {v['name']}: default={v.get('default', 'N/A')}, min={v.get('min', 'N/A')}, max={v.get('max', 'N/A')}")

    def get_detected_variables(self):
        return self._detected_vars

    def get_aedt_info(self):
        project_path = ""
        if self.oProject:
            try:
                project_path = self.oProject.GetPath()
            except Exception:
                pass
        design_name = self.design_combo.currentText()
        return project_path, design_name
