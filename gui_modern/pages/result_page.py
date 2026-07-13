from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFileDialog,
    QPushButton,
)


class ResultPage(QWidget):
    def __init__(self):
        super().__init__()
        self.r2_history = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = SectionTitle("Results")
        layout.addWidget(title)

        chart_container = ChartCard()
        try:
            import pyqtgraph as pg
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setBackground("#1e1e1e")
            self.plot_widget.setLabel("bottom", "Step")
            self.plot_widget.setLabel("left", "Val R2")
            self.plot_widget.addLegend()
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
            chart_container.layout.addWidget(self.plot_widget)
        except ImportError:
            fallback = QLabel("pyqtgraph not installed. Install with: pip install pyqtgraph")
            fallback.setStyleSheet("color: #ff9800; padding: 40px;")
            fallback.setAlignment(Qt.AlignCenter)
            chart_container.layout.addWidget(fallback)
            self.plot_widget = None

        layout.addWidget(chart_container, stretch=2)

        table_container = TableCard()
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(4)
        self.metrics_table.setHorizontalHeaderLabels(["Step", "Val R2", "Status", "Samples"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        table_container.layout.addWidget(self.metrics_table)
        layout.addWidget(table_container, stretch=1)

        export_btn = QPushButton("Export Results")
        export_btn.setFixedHeight(40)
        export_btn.clicked.connect(self._export_results)
        layout.addWidget(export_btn)

    def update_r2_history(self, r2_history):
        self.r2_history = r2_history
        self._update_plot()
        self._update_table()

    def _update_plot(self):
        if not self.plot_widget or not self.r2_history:
            return

        import pyqtgraph as pg
        import math

        valid_r2 = []
        for i, r in enumerate(self.r2_history):
            if isinstance(r, float) and math.isnan(r):
                valid_r2.append(None)
            else:
                valid_r2.append(float(r))

        steps = list(range(1, len(self.r2_history) + 1))
        self.plot_widget.clear()

        vx = [s for s, v in zip(steps, valid_r2) if v is not None]
        vy = [v for v in valid_r2 if v is not None]
        if vx and vy:
            scatter = pg.ScatterPlotItem(
                vx, vy,
                pen=pg.mkPen((86, 156, 214), width=2),
                symbol="o", size=8,
                brush=pg.mkBrush((86, 156, 214))
            )
            self.plot_widget.addItem(scatter)

            line = pg.PlotCurveItem(
                vx, vy, pen=pg.mkPen((86, 156, 214), width=2)
            )
            self.plot_widget.addItem(line)

    def _update_table(self):
        import math
        self.metrics_table.setRowCount(len(self.r2_history))
        for i, r in enumerate(self.r2_history):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            if isinstance(r, float) and math.isnan(r):
                self.metrics_table.setItem(i, 1, QTableWidgetItem("N/A"))
            else:
                self.metrics_table.setItem(i, 1, QTableWidgetItem(f"{r:.4f}"))
            self.metrics_table.setItem(i, 2, QTableWidgetItem("Completed"))
            cumulative = (i + 1) * 15
            self.metrics_table.setItem(i, 3, QTableWidgetItem(str(cumulative)))

    def _export_results(self):
        if not self.r2_history:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "CSV Files (*.csv);;JSON Files (*.json)"
        )
        if not path:
            return

        import csv as csv_module
        import math

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv_module.writer(f)
            writer.writerow(["Step", "Val R2"])
            for i, r in enumerate(self.r2_history):
                r2_str = (
                    f"{r:.4f}"
                    if not (isinstance(r, float) and math.isnan(r))
                    else "N/A"
                )
                writer.writerow([i + 1, r2_str])


class SectionTitle(QWidget):
    def __init__(self, text):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)
        label = QLabel(text)
        label.setStyleSheet("font-size: 22px; font-weight: bold; color: #569cd6;")
        layout.addWidget(label)


class ChartCard(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("R2 History")
        title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 8px; color: #569cd6;")
        self.layout.addWidget(title)


class TableCard(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Metrics Table")
        title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 4px; color: #569cd6;")
        self.layout.addWidget(title)
