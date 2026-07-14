import os
import time
import threading

from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QTextBrowser,
)


class _WorkerThread(QThread):
    def __init__(self, worker):
        super().__init__()
        self.worker = worker

    def run(self):
        self.worker.run()


class MonitorPanel(QFrame):
    started = pyqtSignal(dict, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QGridLayout(self)
        layout.setSpacing(6)
        row = 0

        # Step label
        self.step_label = QLabel("Ready")
        self.step_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.step_label, row, 0, 1, 3)
        row += 1

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar, row, 0, 1, 3)
        row += 1

        # Sweep progress
        sweep_layout_row = row
        self.sweep_label = QLabel("Sweep: waiting")
        self.sweep_label.setStyleSheet("color: gray; font-size: 12px;")
        layout.addWidget(self.sweep_label, row, 0)
        self.sweep_progress_bar = QProgressBar()
        self.sweep_progress_bar.setValue(0)
        self.sweep_progress_bar.setFixedHeight(22)
        layout.addWidget(self.sweep_progress_bar, row, 1, 1, 2)
        row += 1

        # Buttons
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self._on_start)
        layout.addWidget(self.start_btn, row, 0)

        self.abort_btn = QPushButton("Abort")
        self.abort_btn.setEnabled(False)
        self.abort_btn.clicked.connect(self._on_abort)
        layout.addWidget(self.abort_btn, row, 1)

        row += 1

        # Log area
        self.log_browser = QTextBrowser()
        self.log_browser.setFontFamily("Consolas")
        layout.addWidget(self.log_browser, row, 0, 1, 3)

    def set_config(self, config_dict):
        self._config = config_dict

    def set_aedt_info(self, aedt_path, design_name):
        self._aedt_path = aedt_path
        self._design_name = design_name

    def _on_start(self):
        if not hasattr(self, "_config"):
            self.log_browser.append("Error: No configuration set.")
            return

        config = self._config.copy()
        config["design_name"] = self._design_name
        config["aedt_project_path"] = self._aedt_path
        config["project_name"] = time.strftime("%Y%m%d_%H%M%S")

        project_dir = os.path.dirname(self._aedt_path) if os.path.isfile(self._aedt_path) else self._aedt_path
        runs_dir = os.path.join(project_dir, "HUDS_runs")
        run_dir = os.path.join(runs_dir, config["project_name"])
        os.makedirs(run_dir, exist_ok=True)

        from gui_tabs.worker import HUDSWorker

        self._worker = HUDSWorker(config, run_dir, self._aedt_path, self._design_name)

        self._thread = _WorkerThread(self._worker)
        self._worker.signals.log_signal.connect(self._on_log)
        self._worker.signals.progress_signal.connect(self._on_progress)
        self._worker.signals.step_signal.connect(self._on_step)
        self._worker.signals.r2_signal.connect(self._on_r2)
        self._worker.signals.finished_signal.connect(self._on_finished)
        self._worker.signals.sweep_progress_signal.connect(self._on_sweep_progress)
        self._thread.finished.connect(self._on_thread_finished)

        self.start_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)
        self.log_browser.clear()
        self.progress_bar.setValue(0)
        self.sweep_progress_bar.setValue(0)
        self.sweep_label.setText("Sweep: waiting")
        self.sweep_label.setStyleSheet("color: gray; font-size: 12px;")

        self._thread.start()

    def _on_abort(self):
        if hasattr(self, "_worker"):
            self._worker.abort = True
            self.log_browser.append("\n*** Abort requested ***\n")

    def _on_log(self, message):
        self.log_browser.append(message)

    def _on_progress(self, value):
        self.progress_bar.setValue(value)

    def _on_step(self, text):
        self.step_label.setText(text)

    def _on_sweep_progress(self, event_type, data):
        if event_type == "started":
            self.sweep_progress_bar.setRange(0, 0)
            self.sweep_label.setText("仿真运行中...")
            self.sweep_label.setStyleSheet("color: blue; font-size: 12px;")
        elif event_type == "completed":
            self.sweep_progress_bar.setRange(0, 100)
            self.sweep_progress_bar.setValue(100)
            self.sweep_label.setText("仿真完成")
            self.sweep_label.setStyleSheet("color: green; font-size: 12px;")
        elif event_type == "failed":
            self.sweep_progress_bar.setRange(0, 100)
            self.sweep_progress_bar.setValue(0)
            self.sweep_label.setText(f"仿真失败: {data}")
            self.sweep_label.setStyleSheet("color: red; font-size: 12px;")

    def _on_r2(self, r2):
        import math
        if not math.isnan(r2):
            self.log_browser.append(f"  R2 signal: {r2:.4f}")

        main_window = self.window()
        if hasattr(main_window, "result_panel"):
            main_window.result_panel.update_r2(r2)

    def _on_finished(self, success, message):
        self.start_btn.setEnabled(True)
        self.abort_btn.setEnabled(False)
        style = "color: green;" if success else "color: red;"
        self.step_label.setText(f"{'Success' if success else 'Failed'}")
        self.step_label.setStyleSheet(f"{style} font-size: 14px; font-weight: bold;")
        self.log_browser.append(f"\n*** {message} ***\n")

    def _on_thread_finished(self):
        self.start_btn.setEnabled(True)
        self.abort_btn.setEnabled(False)
