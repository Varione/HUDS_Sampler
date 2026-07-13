import os
from PyQt5.QtWidgets import (
    QWizardPage,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFileDialog,
)
import pyqtgraph as pg


class ResultPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("训练结果")
        self.setSubTitle("查看 R2 曲线和每步指标")
        self._r2_history = []
        self._metrics_rows = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(QLabel("R2 变化曲线:"))
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setTitle("R2 per Step")
        self.plot_widget.setLabel("bottom", "Step")
        self.plot_widget.setLabel("left", "R2")
        self.plot_widget.addLegend()
        self.r2_curve = self.plot_widget.plot(name="R2", pen="b", symbol="o", symbolPen="b", symbolBrush="w")
        layout.addWidget(self.plot_widget, 1)

        layout.addWidget(QLabel("每步指标:"))
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Step", "R2", "RMSE", "状态"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table, 1)

        btn_layout = QHBoxLayout()
        self.export_btn = QPushButton("导出 CSV")
        self.export_btn.clicked.connect(self._export_csv)
        btn_layout.addStretch()
        btn_layout.addWidget(self.export_btn)
        layout.addLayout(btn_layout)

    def initializePage(self):
        wizard = self.window()
        self._r2_history = wizard.property("r2_history") or []
        self._update_plot()

    def _update_plot(self):
        if self._r2_history:
            steps = list(range(1, len(self._r2_history) + 1))
            self.r2_curve.setData(steps, self._r2_history)
            self.plot_widget.setXRange(0.5, len(self._r2_history) + 0.5)

        self.table.setRowCount(len(self._r2_history))
        import math
        for i, r2 in enumerate(self._r2_history):
            step_item = QTableWidgetItem(str(i + 1))
            r2_str = f"{r2:.4f}" if not (isinstance(r2, float) and math.isnan(r2)) else "N/A"
            r2_item = QTableWidgetItem(r2_str)
            rmse_item = QTableWidgetItem("-")
            status_item = QTableWidgetItem("完成")
            self.table.setItem(i, 0, step_item)
            self.table.setItem(i, 1, r2_item)
            self.table.setItem(i, 2, rmse_item)
            self.table.setItem(i, 3, status_item)

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 CSV", "", "CSV Files (*.csv)"
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write("Step,R2,RMSE,Status\n")
            import math
            for i, r2 in enumerate(self._r2_history):
                r2_str = f"{r2:.4f}" if not (isinstance(r2, float) and math.isnan(r2)) else "N/A"
                f.write(f"{i+1},{r2_str},-,完成\n")

    def set_next_id(self, nid):
        self._next_id = nid

    def nextId(self):
        return getattr(self, "_next_id", -1)
