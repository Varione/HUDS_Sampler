import csv
import os

import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
)


class ResultPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.r2_values = []
        self._metrics_per_step = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QGridLayout(self)
        layout.setSpacing(6)
        row = 0

        # R2 Plot
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setTitle("R2 Curve")
        self.plot_widget.setLabel("left", "R2")
        self.plot_widget.setLabel("bottom", "Step")
        self.plot_widget.addLegend()
        self.plot_widget.showGrid(x=True, y=True)
        self.r2_curve = self.plot_widget.plot(
            pen=pg.mkPen(color="b", width=2), symbol="o", symbolSize=6, name="val_r2_avg"
        )
        layout.addWidget(self.plot_widget, row, 0, 1, 3)
        row += 1

        # Metrics table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Step", "Val R2 Avg", "N Selected", "Status"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, row, 0, 1, 3)
        row += 1

        # Export button
        btn_layout = QHBoxLayout()
        self.export_btn = QPushButton("Export CSV")
        self.export_btn.clicked.connect(self._on_export)
        btn_layout.addWidget(self.export_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout, row, 0, 1, 3)

    def update_r2(self, r2_value):
        import math
        self.r2_values.append(r2_value)
        steps = list(range(1, len(self.r2_values) + 1))
        values = []
        for v in self.r2_values:
            if math.isnan(v):
                values.append(None)
            else:
                values.append(v)
        self.r2_curve.setData(steps, values)

        row_count = self.table.rowCount()
        self.table.insertRow(row_count)
        step_idx = len(self.r2_values)
        self.table.setItem(row_count, 0, QTableWidgetItem(str(step_idx)))
        if math.isnan(r2_value):
            self.table.setItem(row_count, 1, QTableWidgetItem("N/A"))
        else:
            self.table.setItem(row_count, 1, QTableWidgetItem(f"{r2_value:.4f}"))
        self.table.setItem(row_count, 2, QTableWidgetItem(""))
        self.table.setItem(row_count, 3, QTableWidgetItem("Done"))

    def update_step_info(self, n_selected):
        row_count = self.table.rowCount()
        if row_count > 0:
            item = self.table.item(row_count - 1, 2)
            if item:
                item.setText(str(n_selected))

    def _on_export(self):
        if not self.r2_values:
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "CSV Files (*.csv)"
        )
        if not file_path:
            return

        import math
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Step", "Val R2 Avg"])
            for i, r2 in enumerate(self.r2_values, 1):
                if math.isnan(r2):
                    writer.writerow([i, "N/A"])
                else:
                    writer.writerow([i, f"{r2:.6f}"])
