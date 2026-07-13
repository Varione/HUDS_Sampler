import time
from PyQt5.QtWidgets import (
    QWizardPage,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
)


class AEDTPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("AEDT 连接与项目选择")
        self.setSubTitle("连接到 Ansys Electronics Desktop，选择项目和设计")
        self._oApp = None
        self._oDesktop = None
        self._oProject = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        status_layout = QHBoxLayout()
        self.status_label = QLabel("未连接")
        self.status_label.setStyleSheet("color: gray;")
        self.connect_btn = QPushButton("连接 AEDT")
        self.connect_btn.clicked.connect(self._connect_aedt)
        status_layout.addWidget(self.status_label, 1)
        status_layout.addWidget(self.connect_btn)
        layout.addLayout(status_layout)

        layout.addWidget(QLabel("已打开的项目:"))
        self.project_list = QListWidget()
        self.project_list.currentRowChanged.connect(self._on_project_selected)
        layout.addWidget(self.project_list)

        layout.addWidget(QLabel("设计列表:"))
        self.design_list = QListWidget()
        self.design_list.currentRowChanged.connect(self._on_design_selected)
        layout.addWidget(self.design_list)

        layout.addStretch()

    def _connect_aedt(self):
        self.status_label.setText("正在连接...")
        self.connect_btn.setEnabled(False)
        self.window().setProperty("aedt_connected", False)

        try:
            from win32com.client import Dispatch

            prog_ids = [
                "Ansoft.ElectronicsDesktop.2021.1",
                "Ansoft.ElectronicsDesktop.2022.1",
                "Ansoft.ElectronicsDesktop.2023.1",
                "Ansoft.ElectronicsDesktop.2024.1",
                "Ansoft.ElectronicsDesktop.2025.1",
            ]

            for prog_id in prog_ids:
                try:
                    oApp = Dispatch(prog_id)
                    oDesktop = oApp.GetAppDesktop()
                    version = oDesktop.GetVersion()
                    self._oApp = oApp
                    self._oDesktop = oDesktop
                    self.status_label.setText(f"已连接 AEDT v{version}")
                    self.status_label.setStyleSheet("color: green;")
                    self.window().setProperty("aedt_connected", True)
                    self._populate_projects()
                    self.connect_btn.setEnabled(True)
                    return
                except Exception:
                    continue

            from huds_app.interface.maxwell_sweep import ensure_aedt_running

            self.status_label.setText("AEDT 未运行，正在自动启动...")
            try:
                ensure_aedt_running()
            except FileNotFoundError as e:
                QMessageBox.critical(self, "错误", str(e))
                self.status_label.setText("启动失败")
                self.status_label.setStyleSheet("color: red;")
                self.connect_btn.setEnabled(True)
                return

            for prog_id in prog_ids:
                try:
                    oApp = Dispatch(prog_id)
                    oDesktop = oApp.GetAppDesktop()
                    version = oDesktop.GetVersion()
                    self._oApp = oApp
                    self._oDesktop = oDesktop
                    self.status_label.setText(f"已连接 AEDT v{version}")
                    self.status_label.setStyleSheet("color: green;")
                    self.window().setProperty("aedt_connected", True)
                    self._populate_projects()
                    self.connect_btn.setEnabled(True)
                    return
                except Exception:
                    continue

            self.status_label.setText("连接失败，请手动启动 AEDT 后重试")
            self.status_label.setStyleSheet("color: red;")
            self.connect_btn.setEnabled(True)

        except Exception as e:
            self.status_label.setText(f"错误: {e}")
            self.status_label.setStyleSheet("color: red;")
            self.connect_btn.setEnabled(True)

    def _populate_projects(self):
        self.project_list.clear()
        projects = self._oDesktop.GetProjects()
        for i in range(projects.Count):
            proj = projects(i)
            name = proj.GetName()
            try:
                path = proj.GetPath()
            except Exception:
                path = "未知路径"
            item = QListWidgetItem(f"{name} ({path})")
            item.setData(1, i)
            self.project_list.addItem(item)

    def _on_project_selected(self, row):
        if row < 0 or not self._oDesktop:
            return
        self.design_list.clear()
        item = self.project_list.item(row)
        idx = item.data(1)
        self._oProject = self._oDesktop.GetProjects()(idx)
        designs = self._oProject.GetDesigns()
        for i in range(designs.Count):
            dsgn = designs(i)
            self.design_list.addItem(dsgn.GetName())

    def _on_design_selected(self, row):
        if row < 0 or not self._oProject:
            return
        design_item = self.design_list.item(row)
        design_name = design_item.text()
        try:
            oDesign = self._oProject.SetActiveDesign(design_name)
        except Exception:
            return

        detected_vars = []
        try:
            oModule = oDesign.GetModule("DesignData")
            var_names_list = oModule.GetVariableNames()
            if hasattr(var_names_list, '__iter__'):
                for vname in list(var_names_list):
                    info = {"name": str(vname)}
                    try:
                        info["value"] = str(oModule.GetVariableValue(vname))
                    except Exception:
                        info["value"] = ""
                    try:
                        info["unit"] = str(oModule.GetVariableUnit(str(vname)))
                    except Exception:
                        info["unit"] = ""
                    detected_vars.append(info)
        except Exception:
            pass

        wizard = self.window()
        wizard.setProperty("detected_variables", detected_vars)

    def initializePage(self):
        config = self.window().property("config")
        if config and config.get("aedt_project_path"):
            aedt_path = config["aedt_project_path"]
            self._oDesktop = self.window().property("_oDesktop")
            self._oApp = self.window().property("_oApp")

    def validatePage(self):
        design_item = self.design_list.currentItem()
        if not design_item:
            QMessageBox.warning(self, "警告", "请选择一个设计")
            return False
        design_name = design_item.text()
        wizard = self.window()
        wizard.setProperty("design_name", design_name)
        wizard.setProperty("_oApp", self._oApp)
        wizard.setProperty("_oDesktop", self._oDesktop)
        if self._oProject:
            try:
                wizard.setProperty("aedt_path", self._oProject.GetPath())
            except Exception:
                config = wizard.property("config")
                if config:
                    wizard.setProperty("aedt_path", config.get("aedt_project_path", ""))
        return True

    def set_next_id(self, nid):
        self._next_id = nid

    def nextId(self):
        return getattr(self, "_next_id", -1)
