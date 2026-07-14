import os

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QFileDialog,
    QGroupBox,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QComboBox,
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


class AEDTPage(QWidget):
    def __init__(self):
        super().__init__()
        self._oApp = None
        self._oDesktop = None
        self._oProject = None
        self._aedt_path = ""
        self._design_name = ""
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = SectionTitle("AEDT Connection")
        layout.addWidget(title)

        connect_btn = QPushButton("Connect to AEDT")
        connect_btn.setFixedHeight(48)
        connect_btn.clicked.connect(self._connect_aedt)
        layout.addWidget(connect_btn)

        instance_layout = QHBoxLayout()
        instance_layout.addWidget(QLabel("Instance:"))
        self.instance_combo = QComboBox()
        self.instance_combo.setVisible(False)
        self.instance_combo.currentIndexChanged.connect(self._on_instance_selected)
        instance_layout.addWidget(self.instance_combo, 1)
        layout.addLayout(instance_layout)

        self.status_label = QLabel("Not connected")
        self.status_label.setStyleSheet("color: #888; font-size: 13px; padding: 8px 0;")
        layout.addWidget(self.status_label)

        project_group = QGroupBox("Project Selection")
        project_layout = QVBoxLayout()

        open_btn = QPushButton("Open Project File")
        open_btn.clicked.connect(self._open_project)
        project_layout.addWidget(open_btn)

        self.project_path_edit = QLineEdit()
        self.project_path_edit.setReadOnly(True)
        self.project_path_edit.setPlaceholderText("No project selected")
        project_layout.addWidget(self.project_path_edit)

        self.project_list = QListWidget()
        self.project_list.currentRowChanged.connect(self._on_project_selected)
        project_layout.addWidget(self.project_list)

        project_group.setLayout(project_layout)
        layout.addWidget(project_group)

        design_group = QGroupBox("Design Selection")
        design_layout = QVBoxLayout()
        self.design_list = QListWidget()
        self.design_list.currentRowChanged.connect(self._on_design_selected)
        design_layout.addWidget(self.design_list)
        design_group.setLayout(design_layout)
        layout.addWidget(design_group)

        layout.addStretch()

    def _connect_aedt(self):
        try:
            from huds_app.utils.aedt_instances import enumerate_aedt_instances
            from huds_app.interface.maxwell_sweep import ensure_aedt_running

            self.status_label.setText("Connecting...")

            instances = enumerate_aedt_instances()

            if not instances:
                ensure_aedt_running()
                instances = enumerate_aedt_instances()

            if not instances:
                self.status_label.setText("Failed to connect to AEDT")
                self.status_label.setStyleSheet("color: #f44336; font-size: 13px; padding: 8px 0;")
                return

            if len(instances) == 1:
                self._oApp = instances[0]['oApp']
                self._oDesktop = instances[0]['oDesktop']
                self.instance_combo.setVisible(False)
            else:
                self.instance_combo.clear()
                for inst in instances:
                    self.instance_combo.addItem(inst['label'])
                self.instance_combo.setVisible(True)
                self._on_instance_selected(0)
                return

            version = self._oDesktop.GetVersion()
            self.status_label.setText(f"Connected - AEDT {version}")
            self.status_label.setStyleSheet("color: #4caf50; font-size: 13px; padding: 8px 0;")
            self._refresh_projects()

        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            self.status_label.setStyleSheet("color: #f44336; font-size: 13px; padding: 8px 0;")

    def _on_instance_selected(self, index):
        from huds_app.utils.aedt_instances import enumerate_aedt_instances
        instances = enumerate_aedt_instances()
        if 0 <= index < len(instances):
            self._oApp = instances[index]['oApp']
            self._oDesktop = instances[index]['oDesktop']
            version = instances[index]['version']
            self.status_label.setText(f"Connected - AEDT {version}")
            self.status_label.setStyleSheet("color: #4caf50; font-size: 13px; padding: 8px 0;")
            self._refresh_projects()

    def _refresh_projects(self):
        self.project_list.clear()
        if not self._oDesktop:
            return
        projects = self._oDesktop.GetProjects()
        for i in range(projects.Count):
            proj = projects(i)
            name = proj.GetName()
            try:
                path = proj.GetPath()
            except Exception:
                path = "Unknown"
            self.project_list.addItem(f"{name}  ({path})")

    def _open_project(self):
        if not self._oDesktop:
            self.status_label.setText("Connect to AEDT first")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Select AEDT Project", "", "AEDT Files (*.aedt)"
        )
        if not path:
            return

        try:
            projects = self._oDesktop.GetProjects()
            for i in range(projects.Count):
                try:
                    self._oDesktop.CloseProject(projects(i).GetName())
                except Exception:
                    pass

            import time
            time.sleep(2)
            self._oProject = self._oDesktop.OpenProject(path)
            time.sleep(3)

            self._aedt_path = path
            self.project_path_edit.setText(path)
            self._refresh_designs()
        except Exception as e:
            self.status_label.setText(f"Error opening project: {e}")

    def _on_project_selected(self, row):
        if not self._oDesktop:
            return
        try:
            projects = self._oDesktop.GetProjects()
            if 0 <= row < projects.Count:
                self._oProject = projects(row)
                self._aedt_path = self._oProject.GetPath()
                self.project_path_edit.setText(self._aedt_path)
                self._refresh_designs()
        except Exception:
            pass

    def _refresh_designs(self):
        self.design_list.clear()
        if not self._oProject:
            return
        try:
            designs = self._oProject.GetDesigns()
            for i in range(designs.Count):
                self.design_list.addItem(designs(i).GetName())
        except Exception:
            pass

    def _on_design_selected(self, row):
        if row < 0:
            return
        design_name = self.design_list.item(row).text()
        print(f"[AEDT] Design selected: {design_name}")

        # Try COM connection first
        if self._oProject:
            try:
                oDesign = self._oProject.SetActiveDesign(design_name)
            except Exception as e:
                print(f"[AEDT] SetActiveDesign failed: {e}")

        # Auto-detect variables from .aedt file
        aedt_path = getattr(self, "_aedt_file_path", "") or ""
        if not aedt_path and self._oProject:
            try:
                aedt_path = self._oProject.GetPath()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"获取项目路径失败: {e}")

        # Ensure aedt_path points to .aedt file, not directory
        if aedt_path and os.path.isdir(aedt_path):
            import glob as g
            aedt_files = g.glob(os.path.join(aedt_path, "*.aedt"))
            matched_file = _match_aedt_by_project(aedt_files, self._oProject)
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
                f"未找到 .aedt 路径\nfile_path={getattr(self, '_aedt_file_path', 'NOT SET')}\nhas_project={self._oProject is not None}")

        self._detected_vars = detected_vars
        self._detected_outputs = detected_outputs

    def get_detected_variables(self):
        return getattr(self, '_detected_vars', [])

    def get_detected_outputs(self):
        return getattr(self, '_detected_outputs', [])

    def get_aedt_info(self):
        return self._aedt_path, self._design_name


class SectionTitle(QWidget):
    def __init__(self, text):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)
        label = QLabel(text)
        label.setStyleSheet("font-size: 22px; font-weight: bold; color: #569cd6;")
        layout.addWidget(label)
