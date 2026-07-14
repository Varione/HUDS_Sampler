import os
from PyQt5.QtWidgets import (
    QWizardPage,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QTextBrowser,
)


class MonitorPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("训练监控")
        self.setSubTitle("开始主动学习训练循环，实时监控进度")
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        step_layout = QHBoxLayout()
        step_layout.addWidget(QLabel("当前步骤:"))
        self.step_label = QLabel("未开始")
        step_layout.addWidget(self.step_label, 1)
        layout.addLayout(step_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.log_browser = QTextBrowser()
        self.log_browser.setMinimumHeight(300)
        layout.addWidget(self.log_browser)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始训练")
        self.start_btn.clicked.connect(self._start_training)
        self.abort_btn = QPushButton("中止")
        self.abort_btn.setEnabled(False)
        self.abort_btn.clicked.connect(self._abort_training)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.abort_btn)
        layout.addLayout(btn_layout)

    def _start_training(self):
        from gui_wizard.worker import HUDSWorker

        wizard = self.window()
        config = wizard.property("config")
        if not config:
            self.log_browser.append("错误: 未找到配置，请先完成配置页面")
            return

        aedt_path = wizard.property("aedt_path") or config.get("aedt_project_path", "")
        design_name = wizard.property("design_name") or ""

        if not aedt_path:
            self.log_browser.append("错误: 未设置 AEDT 项目路径")
            return
        if not design_name:
            self.log_browser.append("错误: 未选择设计名称")
            return

        config["aedt_project_path"] = aedt_path
        config["design_name"] = design_name

        # aedt_path is the .aedt file path; project_dir is its containing folder
        project_dir = os.path.dirname(aedt_path) if os.path.isfile(aedt_path) else aedt_path
        runs_dir = os.path.join(project_dir, "HUDS_runs")
        run_dir = os.path.join(runs_dir, config["project_name"])
        os.makedirs(run_dir, exist_ok=True)
        config_path = os.path.join(run_dir, "config.json")

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        self.log_browser.append(f"配置已保存到: {config_path}")

        self._worker = HUDSWorker(config, run_dir, aedt_path, design_name)
        self._worker.progress_signal.connect(self.progress_bar.setValue)
        self._worker.log_signal.connect(self.log_browser.append)
        self._worker.step_signal.connect(self.step_label.setText)
        self._worker.r2_signal.connect(self._on_r2)
        self._worker.finished_signal.connect(self._on_finished)

        self.start_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)
        self._worker.start()

    def _abort_training(self):
        if self._worker:
            self._worker.abort = True
            self.log_browser.append("已请求中止训练...")

    def _on_r2(self, r2_val):
        wizard = self.window()
        r2_history = wizard.property("r2_history") or []
        r2_history.append(r2_val)
        wizard.setProperty("r2_history", r2_history)

    def _on_finished(self, success, message):
        self.start_btn.setEnabled(True)
        self.abort_btn.setEnabled(False)
        self.log_browser.append(f"\n训练{'成功' if success else '失败'}: {message}")

    def set_next_id(self, nid):
        self._next_id = nid

    def nextId(self):
        return getattr(self, "_next_id", -1)

    def initializePage(self):
        self.progress_bar.setValue(0)
        self.step_label.setText("未开始")
        self.log_browser.clear()
