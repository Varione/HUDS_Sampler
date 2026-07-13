import time
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QWidget,
)


class ConfigSignals(QObject):
    config_ready = pyqtSignal()


class ConfigPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._signals = ConfigSignals()
        self._setup_ui()

    @property
    def config_ready_signal(self):
        return self._signals.config_ready

    def _setup_ui(self):
        layout = QGridLayout(self)
        layout.setSpacing(6)
        row = 0

        def add_label(text, col=0):
            lbl = QLabel(text)
            layout.addWidget(lbl, row, col, Qt.AlignLeft | Qt.AlignVCenter)
            return lbl

        def add_widget(widget, col=1, stretch=0):
            layout.addWidget(widget, row, col, 1, stretch)

        # AEDT project path
        add_label("AEDT Project Path (.aedt)")
        self.aedt_path_edit = QLineEdit()
        self.aedt_path_edit.setPlaceholderText("E:\\project\\model.aedt")
        add_widget(self.aedt_path_edit, stretch=2)

        # Variable selection table
        add_label("Design Variables (select):", col=0)
        self.var_table = QTableWidget(0, 6)
        self.var_table.setHorizontalHeaderLabels(["Select", "Name", "Default", "Min", "Max", "Unit"])
        self.var_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.var_table.setColumnHidden(5, True)
        add_widget(self.var_table, col=1, stretch=2)

        # Output names
        add_label("Output Names (comma separated)")
        self.output_names_edit = QLineEdit("peak_force_y,peak_force_z")
        add_widget(self.output_names_edit, stretch=2)

        # Steady state pct
        add_label("Steady State Pct")
        self.steady_pct_spin = QDoubleSpinBox()
        self.steady_pct_spin.setRange(0.01, 1.0)
        self.steady_pct_spin.setValue(0.2)
        self.steady_pct_spin.setSingleStep(0.05)
        add_widget(self.steady_pct_spin)

        # Total samples
        add_label("Total Samples")
        self.total_samples_spin = QSpinBox()
        self.total_samples_spin.setRange(10, 100000)
        self.total_samples_spin.setValue(600)
        self.total_samples_spin.setSingleStep(50)
        add_widget(self.total_samples_spin)

        # Initial train size
        add_label("Initial Train Size")
        self.initial_train_spin = QSpinBox()
        self.initial_train_spin.setRange(5, 10000)
        self.initial_train_spin.setValue(20)
        self.initial_train_spin.setSingleStep(5)
        add_widget(self.initial_train_spin)

        # Sample per step
        add_label("Sample Per Step")
        self.sample_per_step_spin = QSpinBox()
        self.sample_per_step_spin.setRange(1, 5000)
        self.sample_per_step_spin.setValue(15)
        self.sample_per_step_spin.setSingleStep(5)
        add_widget(self.sample_per_step_spin)

        # Max steps
        add_label("Max Steps")
        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(1, 100)
        self.max_steps_spin.setValue(3)
        self.max_steps_spin.setSingleStep(1)
        add_widget(self.max_steps_spin)

        # Epochs per step
        add_label("Epochs Per Step")
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 10000)
        self.epochs_spin.setValue(200)
        self.epochs_spin.setSingleStep(50)
        add_widget(self.epochs_spin)

        # Device
        add_label("Device")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda"])
        self.device_combo.setCurrentText("cpu")
        add_widget(self.device_combo)

    def set_detected_variables(self, detected_vars):
        self.var_table.setRowCount(len(detected_vars))
        for i, var in enumerate(detected_vars):
            cb = QCheckBox()
            cb.setChecked(True)
            cb_widget = QWidget()
            cb_layout = QHBoxLayout(cb_widget)
            cb_layout.addWidget(cb)
            cb_layout.setAlignment(cb, Qt.AlignCenter)
            cb_layout.setContentsMargins(0, 0, 0, 0)
            self.var_table.setCellWidget(i, 0, cb_widget)

            name_item = QTableWidgetItem(var.get("name", ""))
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.var_table.setItem(i, 1, name_item)

            default_item = QTableWidgetItem(var.get("default", ""))
            default_item.setFlags(default_item.flags() & ~Qt.ItemIsEditable)
            self.var_table.setItem(i, 2, default_item)

            min_item = QTableWidgetItem("")
            self.var_table.setItem(i, 3, min_item)

            max_item = QTableWidgetItem("")
            self.var_table.setItem(i, 4, max_item)

            unit_item = QTableWidgetItem(var.get("unit", ""))
            unit_item.setFlags(unit_item.flags() & ~Qt.ItemIsEditable)
            self.var_table.setItem(i, 5, unit_item)

    def get_config(self):
        from huds_app.utils.aedt_parser import parse_value_with_unit
        
        selected_vars = []
        for i in range(self.var_table.rowCount()):
            cb_widget = self.var_table.cellWidget(i, 0)
            cb = cb_widget.findChild(QCheckBox) if cb_widget else None
            if cb and cb.isChecked():
                name_item = self.var_table.item(i, 1)
                min_item = self.var_table.item(i, 3)
                max_item = self.var_table.item(i, 4)
                unit_item = self.var_table.item(i, 5)
                
                var_name = name_item.text() if name_item else ""
                min_str = min_item.text() if min_item else ""
                max_str = max_item.text() if max_item else ""
                var_unit = unit_item.text() if unit_item else ""
                
                min_num, _ = parse_value_with_unit(min_str)
                max_num, _ = parse_value_with_unit(max_str)
                
                selected_vars.append({
                    "name": var_name,
                    "min": float(min_num) if min_num is not None else 0.0,
                    "max": float(max_num) if max_num is not None else 1.0,
                    "sample_points": 60,
                    "unit": var_unit,
                })

        output_names = [x.strip() for x in self.output_names_edit.text().split(",") if x.strip()]

        return {
            "project_name": time.strftime("%Y%m%d_%H%M%S"),
            "random_seed": 42,
            "aedt_project_path": self.aedt_path_edit.text().strip(),
            "design_name": "",
            "variables": selected_vars,
            "candidate_pool": {
                "total_samples": self.total_samples_spin.value(),
            },
            "split": {
                "train_split": 0.8,
                "val_split": 0.1,
                "test_split": 0.1,
            },
            "model": {
                "model_type": "vector_to_vector",
                "output_names": output_names,
                "hidden_dim": 64,
                "encoder_blocks": 2,
                "dropout": 0.1,
            },
            "training": {
                "initial_train_size": self.initial_train_spin.value(),
                "sample_per_step": self.sample_per_step_spin.value(),
                "max_steps": self.max_steps_spin.value(),
                "epochs_per_step": self.epochs_spin.value(),
                "batch_size": 32,
                "learning_rate": 0.001,
                "patience": 30,
                "device": self.device_combo.currentText(),
            },
            "huds": {
                "repeat_times": 10,
                "topk_ratio": 0.6,
                "batch_size": 128,
                "use_faiss": False,
            },
            "steady_state_pct": self.steady_pct_spin.value(),
        }
