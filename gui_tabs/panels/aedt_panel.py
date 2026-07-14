import os

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QLabel,
    QMessageBox,
    QPushButton,
)


def _match_aedt_by_project(aedt_files, oProject):
    """Match .aedt file to project name when multiple files exist in directory."""
    if len(aedt_files) == 1:
        return aedt_files[0]
    proj_name = ""
    if oProject:
        try:
            proj_name = oProject.GetName()
        except Exception:
            pass
    if proj_name:
        for af in aedt_files:
            base = os.path.splitext(os.path.basename(af))[0]
            if base == proj_name:
                return af
    return aedt_files[0] if aedt_files else None


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
        self._detected_outputs = []

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
        print(f"[AEDT] Design selected: {design_name}")

        # Try COM connection first
        try:
            oDesign = self.oProject.SetActiveDesign(design_name)
        except Exception as e:
            print(f"[AEDT] SetActiveDesign failed: {e}")

        # Auto-detect variables from .aedt file
        aedt_path = ""
        if self.oProject:
            try:
                aedt_path = self.oProject.GetPath()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"获取项目路径失败: {e}")

        # Ensure aedt_path points to .aedt file, not directory
        if aedt_path and os.path.isdir(aedt_path):
            import glob as g
            aedt_files = g.glob(os.path.join(aedt_path, "*.aedt"))
            matched_file = _match_aedt_by_project(aedt_files, self.oProject)
            if matched_file:
                aedt_path = matched_file
            else:
                QMessageBox.warning(self, "调试",
                    f"目录 {aedt_path} 中未找到 .aedt 文件")

        detected_vars = []
        detected_outputs = []
        if aedt_path:
            try:
                from huds_app.utils.aedt_parser import parse_aedt_variables, parse_aedt_outputs
                detected_vars = parse_aedt_variables(aedt_path, design_name)
                detected_outputs = parse_aedt_outputs(aedt_path, design_name)
                QMessageBox.information(self, "检测结果",
                    f"检测到 {len(detected_vars)} 个变量: {[v['name'] for v in detected_vars]}\n"
                    f"检测到 {len(detected_outputs)} 个输出: {[o['name'] for o in detected_outputs]}")
            except Exception as e:
                import traceback
                QMessageBox.critical(self, "解析错误",
                    f"变量/输出检测失败:\n{str(e)}\n\n{traceback.format_exc()[:500]}")
        else:
            QMessageBox.warning(self, "调试",
                f"未找到 .aedt 路径\nhas_project={self.oProject is not None}")

        self._detected_vars = detected_vars
        self._detected_outputs = detected_outputs

    def get_detected_variables(self):
        return self._detected_vars

    def get_detected_outputs(self):
        return self._detected_outputs

    def get_aedt_info(self):
        project_path = ""
        if self.oProject:
            try:
                project_path = self.oProject.GetPath()
            except Exception:
                pass
        design_name = self.design_combo.currentText()
        return project_path, design_name
