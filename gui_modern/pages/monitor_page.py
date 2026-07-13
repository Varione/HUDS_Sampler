from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QTextBrowser,
    QPushButton,
)


class MonitorPage(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = SectionTitle("Monitor")
        layout.addWidget(title)

        controls = QHBoxLayout()
        self.start_btn = QPushButton("Start Workflow")
        self.start_btn.setFixedHeight(48)
        self.start_btn.clicked.connect(self._start_workflow)
        controls.addWidget(self.start_btn)

        self.abort_btn = QPushButton("Abort")
        self.abort_btn.setFixedHeight(48)
        self.abort_btn.setEnabled(False)
        self.abort_btn.setStyleSheet(
            "background-color: #c62828; color: #fff; border: none; "
            "border-radius: 6px; padding: 10px 24px; font-weight: bold;"
        )
        self.abort_btn.clicked.connect(self._abort_workflow)
        controls.addWidget(self.abort_btn)

        layout.addLayout(controls)

        progress_container = ProgressCard()
        self.progress_label = QLabel("Progress: 0%")
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_container.layout.addWidget(self.progress_label)
        progress_container.layout.addWidget(self.progress_bar)
        layout.addWidget(progress_container)

        log_container = LogCard()
        self.log_browser = QTextBrowser()
        self.log_browser.setFontFamily("Consolas, Courier New, monospace")
        self.log_browser.setStyleSheet(
            "background-color: #1e1e1e; color: #d4d4d4; font-size: 12px; "
            "border: 1px solid #3c3c3c; border-radius: 4px;"
        )
        log_container.layout.addWidget(self.log_browser)
        layout.addWidget(log_container, stretch=1)

    def _log(self, msg):
        self.log_browser.append(msg)

    def _start_workflow(self):
        from gui_modern.window import HUDSModernWindow

        main_window = self.window()
        if not isinstance(main_window, HUDSModernWindow):
            return

        config = main_window.config_page.get_config()
        aedt_path, design_name = main_window.aedt_page.get_aedt_info()

        if not aedt_path:
            self._log("Error: No AEDT project selected. Go to AEDT Connection page.")
            return
        if not design_name:
            self._log("Error: No design selected. Select a design in AEDT Connection page.")
            return

        config["aedt_project_path"] = aedt_path
        config["design_name"] = design_name
        import os
        project_dir = os.path.dirname(aedt_path) if os.path.isfile(aedt_path) else aedt_path

        self._log("Starting HUDS Active Learning workflow...")
        self._log(f"Project: {aedt_path}")
        self._log(f"Design: {design_name}")
        self._log(f"Max steps: {config['training']['max_steps']}")
        self._log("-" * 40)

        from gui_modern.worker import HUDSWorker
        self.worker = HUDSWorker(config, aedt_path, design_name, project_dir)
        self.worker.log_signal.connect(self._log)
        self.worker.progress_signal.connect(self._on_progress)
        self.worker.step_done_signal.connect(self._on_step_done)
        self.worker.finished_all_signal.connect(self._on_finished)
        self.worker.error_signal.connect(self._on_error)

        self.start_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)
        self.worker.start()

    def _abort_workflow(self):
        if self.worker and self.worker.isRunning():
            self.worker.abort()
            self._log("Abort requested, waiting for current operation to complete...")
            self.abort_btn.setEnabled(False)

    def _on_progress(self, value):
        self.progress_bar.setValue(value)
        self.progress_label.setText(f"Progress: {value}%")

    def _on_step_done(self, step, metrics):
        self._log(f"Step {step} completed.")
        r2 = metrics.get("val_r2_avg", float("nan"))
        import math
        r2_str = f"{r2:.4f}" if not (isinstance(r2, float) and math.isnan(r2)) else "N/A"
        self._log(f"  val_r2_avg = {r2_str}")

    def _on_finished(self, r2_history):
        self._log("-" * 40)
        self._log("Workflow completed successfully!")
        self.start_btn.setEnabled(True)
        self.abort_btn.setEnabled(False)

        from gui_modern.window import HUDSModernWindow
        main_window = self.window()
        if isinstance(main_window, HUDSModernWindow):
            main_window.result_page.update_r2_history(r2_history)

    def _on_error(self, msg):
        self._log(f"ERROR: {msg}")
        self.start_btn.setEnabled(True)
        self.abort_btn.setEnabled(False)


class SectionTitle(QWidget):
    def __init__(self, text):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)
        label = QLabel(text)
        label.setStyleSheet("font-size: 22px; font-weight: bold; color: #569cd6;")
        layout.addWidget(label)


class ProgressCard(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)


class LogCard(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Log Output")
        title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 4px; color: #569cd6;")
        self.layout.addWidget(title)
