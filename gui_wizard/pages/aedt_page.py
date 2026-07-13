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
    QComboBox,
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

        instance_layout = QHBoxLayout()
        instance_layout.addWidget(QLabel("实例选择:"))
        self.instance_combo = QComboBox()
        self.instance_combo.setVisible(False)
        self.instance_combo.currentIndexChanged.connect(self._on_instance_selected)
        instance_layout.addWidget(self.instance_combo, 1)
        layout.addLayout(instance_layout)

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
            from huds_app.utils.aedt_instances import enumerate_aedt_instances
            from huds_app.interface.maxwell_sweep import ensure_aedt_running

            instances = enumerate_aedt_instances()

            if not instances:
                self.status_label.setText("AEDT 未运行，正在自动启动...")
                try:
                    ensure_aedt_running()
                except FileNotFoundError as e:
                    QMessageBox.critical(self, "错误", str(e))
                    self.status_label.setText("启动失败")
                    self.status_label.setStyleSheet("color: red;")
                    self.connect_btn.setEnabled(True)
                    return
                instances = enumerate_aedt_instances()

            if not instances:
                self.status_label.setText("连接失败，请手动启动 AEDT 后重试")
                self.status_label.setStyleSheet("color: red;")
                self.connect_btn.setEnabled(True)
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
            self.status_label.setText(f"已连接 AEDT v{version}")
            self.status_label.setStyleSheet("color: green;")
            self.window().setProperty("aedt_connected", True)
            self._populate_projects()
            self.connect_btn.setEnabled(True)

        except Exception as e:
            self.status_label.setText(f"错误: {e}")
            self.status_label.setStyleSheet("color: red;")
            self.connect_btn.setEnabled(True)

    def _on_instance_selected(self, index):
        from huds_app.utils.aedt_instances import enumerate_aedt_instances
        instances = enumerate_aedt_instances()
        if 0 <= index < len(instances):
            self._oApp = instances[index]['oApp']
            self._oDesktop = instances[index]['oDesktop']
            version = instances[index]['version']
            self.status_label.setText(f"已连接 AEDT v{version}")
            self.status_label.setStyleSheet("color: green;")
            self.window().setProperty("aedt_connected", True)
            self._populate_projects()
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
        except Exception as e:
            print(f"[AEDT] SetActiveDesign failed: {e}")
            return

        # Auto-detect variables from .aedt file
        try:
            aedt_path = self._oProject.GetPath()
        except Exception:
            aedt_path = None
        
        if aedt_path:
            from huds_app.utils.aedt_parser import parse_aedt_variables
            vars = parse_aedt_variables(aedt_path, design_name)
            if vars:
                wizard = self.window()
                wizard.setProperty("detected_variables", vars)
                print(f"[AEDT] Detected {len(vars)} variables for design '{design_name}':")
                for v in vars:
                    print(f"  {v['name']}: default={v.get('default', 'N/A')}, min={v.get('min', 'N/A')}, max={v.get('max', 'N/A')}")

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
                pass
        return True

    def set_next_id(self, nid):
        self._next_id = nid

    def nextId(self):
        return getattr(self, "_next_id", -1)
