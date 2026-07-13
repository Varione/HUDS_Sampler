from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QFileDialog,
    QGroupBox,
    QLineEdit,
    QPushButton,
)


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
            from win32com.client import Dispatch
            from huds_app.interface.maxwell_sweep import ensure_aedt_running

            self.status_label.setText("Connecting...")

            prog_ids = [
                "Ansoft.ElectronicsDesktop.2025.1",
                "Ansoft.ElectronicsDesktop.2024.1",
                "Ansoft.ElectronicsDesktop.2023.1",
                "Ansoft.ElectronicsDesktop.2022.1",
                "Ansoft.ElectronicsDesktop.2021.1",
            ]

            for prog_id in prog_ids:
                try:
                    self._oApp = Dispatch(prog_id)
                    self._oDesktop = self._oApp.GetAppDesktop()
                    version = self._oDesktop.GetVersion()
                    self.status_label.setText(f"Connected - AEDT {version}")
                    self.status_label.setStyleSheet("color: #4caf50; font-size: 13px; padding: 8px 0;")
                    self._refresh_projects()
                    return
                except Exception:
                    continue

            ensure_aedt_running()

            for prog_id in prog_ids:
                try:
                    self._oApp = Dispatch(prog_id)
                    self._oDesktop = self._oApp.GetAppDesktop()
                    version = self._oDesktop.GetVersion()
                    self.status_label.setText(f"Connected - AEDT {version}")
                    self.status_label.setStyleSheet("color: #4caf50; font-size: 13px; padding: 8px 0;")
                    self._refresh_projects()
                    return
                except Exception:
                    continue

            self.status_label.setText("Failed to connect to AEDT")
            self.status_label.setStyleSheet("color: #f44336; font-size: 13px; padding: 8px 0;")

        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            self.status_label.setStyleSheet("color: #f44336; font-size: 13px; padding: 8px 0;")

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
        if self.design_list.count() > 0:
            self._design_name = self.design_list.item(row).text()

        detected_vars = []
        if self._oProject:
            try:
                oDesign = self._oProject.SetActiveDesign(self._design_name)
                var_names_list = oDesign.GetVariableNames()
            except Exception:
                try:
                    oDesign = self._oProject.SetActiveDesign(self._design_name)
                    oModule = oDesign.GetModule("DesignProperties")
                    var_names_list = oModule.GetVariableNames()
                except Exception:
                    var_names_list = []

            if hasattr(var_names_list, '__iter__'):
                for vname in list(var_names_list):
                    info = {"name": str(vname)}
                    try:
                        info["value"] = str(oDesign.GetVariableValue(str(vname)))
                    except Exception:
                        info["value"] = ""
                    detected_vars.append(info)

        self._detected_vars = detected_vars

    def get_detected_variables(self):
        return getattr(self, '_detected_vars', [])

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
