import os
from PyQt5.QtWidgets import (
    QWizardPage,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QFileDialog,
)


class AEDTPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("AEDT 连接与项目选择")
        self.setSubTitle("连接到 Ansys Electronics Desktop 或直接浏览 .aedt 文件")
        self._oApp = None
        self._oDesktop = None
        self._oProject = None
        self._aedt_file_path = ""
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Connection method selector
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("连接方式:"))
        self.connect_btn = QPushButton("连接运行中的 AEDT")
        self.connect_btn.clicked.connect(self._connect_aedt)
        self.browse_btn = QPushButton("浏览 .aedt 文件...")
        self.browse_btn.clicked.connect(self._browse_aedt_file)
        method_layout.addWidget(self.connect_btn)
        method_layout.addWidget(self.browse_btn)
        layout.addLayout(method_layout)

        status_layout = QHBoxLayout()
        self.status_label = QLabel("未连接")
        self.status_label.setStyleSheet("color: gray;")
        status_layout.addWidget(self.status_label, 1)
        layout.addLayout(status_layout)

        # Project list (for COM connection)
        layout.addWidget(QLabel("已打开的项目:"))
        self.project_list = QListWidget()
        self.project_list.currentRowChanged.connect(self._on_project_selected)
        layout.addWidget(self.project_list)

        # AEDT file path display
        self.aedt_path_label = QLabel("")
        self.aedt_path_label.setStyleSheet("color: blue;")
        layout.addWidget(self.aedt_path_label)

        layout.addWidget(QLabel("设计列表:"))
        self.design_list = QListWidget()
        self.design_list.currentRowChanged.connect(self._on_design_selected)
        layout.addWidget(self.design_list)

        layout.addStretch()

    def _connect_aedt(self):
        self.status_label.setText("正在连接...")
        self.connect_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.window().setProperty("aedt_connected", False)

        try:
            from win32com.client import Dispatch
            from huds_app.interface.maxwell_sweep import ensure_aedt_running

            prog_ids = [
                "Ansoft.ElectronicsDesktop.2021.1",
                "Ansoft.ElectronicsDesktop.2022.1",
                "Ansoft.ElectronicsDesktop.2023.1",
                "Ansoft.ElectronicsDesktop.2024.1",
                "Ansoft.ElectronicsDesktop.2025.1",
            ]

            connected = False
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
                    connected = True
                    break
                except Exception:
                    continue

            if not connected:
                self.status_label.setText("AEDT 未运行，正在自动启动...")
                try:
                    ensure_aedt_running()
                except FileNotFoundError as e:
                    QMessageBox.critical(self, "错误", str(e))
                    self.status_label.setText("启动失败，请尝试浏览 .aedt 文件方式")
                    self.status_label.setStyleSheet("color: red;")
                    self.connect_btn.setEnabled(True)
                    self.browse_btn.setEnabled(True)
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
                        connected = True
                        break
                    except Exception:
                        continue

            if not connected:
                self.status_label.setText("连接失败，请使用浏览 .aedt 文件方式")
                self.status_label.setStyleSheet("color: red;")

            self.connect_btn.setEnabled(True)
            self.browse_btn.setEnabled(True)

        except Exception as e:
            self.status_label.setText(f"错误: {e}")
            self.status_label.setStyleSheet("color: red;")
            self.connect_btn.setEnabled(True)
            self.browse_btn.setEnabled(True)

    def _browse_aedt_file(self):
        """Browse for an .aedt file directly - bypasses COM instance limitation."""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 AEDT 项目文件", "", "AEDT Files (*.aedt)"
        )
        if not path:
            return

        self._aedt_file_path = path
        aedt_dir = os.path.dirname(path)
        aedt_name = os.path.splitext(os.path.basename(path))[0]

        self.status_label.setText(f"已选择文件: {os.path.basename(path)}")
        self.status_label.setStyleSheet("color: green;")
        self.aedt_path_label.setText(f"路径: {aedt_dir}")

        # Parse designs from .aedt file
        self.project_list.clear()
        proj_item = QListWidgetItem(f"{aedt_name} ({aedt_dir})")
        proj_item.setData(1, -1)  # -1 indicates file-based, not COM
        self.project_list.addItem(proj_item)

        self._oProject = None
        designs = self._parse_designs_from_aedt(path)
        self.design_list.clear()
        for d in designs:
            self.design_list.addItem(d)

    def _parse_designs_from_aedt(self, aedt_path):
        """Parse design names from .aedt file."""
        try:
            with open(aedt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            designs = []
            for line in lines:
                if "Name='" in line and ("Design.bmp" in ''.join(lines[lines.index(line):lines.index(line)+5]) or
                                          any(c in line for c in ['ly', 'ManagedFiles'])):
                    import re
                    match = re.search(r"Name='([^']*)'", line)
                    if match:
                        designs.append(match.group(1))
            return list(dict.fromkeys(designs))  # Deduplicate preserving order
        except Exception as e:
            print(f"[AEDT] Failed to parse designs: {e}")
            return []

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
        if idx < 0:
            return  # File-based, designs already loaded

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
