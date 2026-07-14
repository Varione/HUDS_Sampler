from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QPushButton,
    QFileDialog,
    QScrollArea,
    QGroupBox,
    QComboBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)


class ConfigPage(QWidget):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(16)

        title = SectionTitle("Configuration")
        layout.addWidget(title)

        project_group = QGroupBox("Project Settings")
        project_layout = QFormLayout()
        self.project_name = QLineEdit()
        self.project_name.setPlaceholderText("Auto-generated timestamp")
        project_layout.addRow("Project Name:", self.project_name)
        self.random_seed = QSpinBox()
        self.random_seed.setValue(42)
        self.random_seed.setMinimum(0)
        self.random_seed.setMaximum(999999)
        project_layout.addRow("Random Seed:", self.random_seed)
        project_group.setLayout(project_layout)
        layout.addWidget(project_group)

        var_group = QGroupBox("Sweep Variables (select from design)")
        var_layout = QVBoxLayout()

        self.var_table = QTableWidget(0, 6)
        self.var_table.setHorizontalHeaderLabels(["Select", "Name", "Default", "Min", "Max", "Unit"])
        self.var_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.var_table.setColumnHidden(5, True)
        var_layout.addWidget(self.var_table)

        var_group.setLayout(var_layout)
        layout.addWidget(var_group)

        output_group = QGroupBox("Output Variables")
        output_layout = QVBoxLayout()

        auto_out_row = QHBoxLayout()
        auto_out_row.addWidget(QLabel("Detected outputs:"))
        self.auto_detect_outputs_btn = QPushButton("Auto Fill")
        self.auto_detect_outputs_btn.clicked.connect(self._auto_fill_outputs)
        auto_out_row.addWidget(self.auto_detect_outputs_btn)
        output_layout.addLayout(auto_out_row)

        self.output_table = QTableWidget(0, 3)
        self.output_table.setHorizontalHeaderLabels(["Select", "Name", "Type"])
        self.output_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        output_layout.addWidget(self.output_table)

        manual_row = QHBoxLayout()
        manual_row.addWidget(QLabel("Manual (comma separated):"))
        self.output_names = QLineEdit("")
        self.output_names.setPlaceholderText("Comma-separated output names")
        manual_row.addWidget(self.output_names, 1)
        output_layout.addLayout(manual_row)

        steady_row = QHBoxLayout()
        steady_row.addWidget(QLabel("Steady State Pct:"))
        self.steady_pct = QDoubleSpinBox()
        self.steady_pct.setValue(0.2)
        self.steady_pct.setRange(0.01, 1.0)
        self.steady_pct.setSingleStep(0.05)
        steady_row.addWidget(self.steady_pct)
        output_layout.addLayout(steady_row)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        train_group = QGroupBox("Training Settings")
        train_layout = QFormLayout()
        self.initial_train = QSpinBox()
        self.initial_train.setValue(20)
        self.initial_train.setMinimum(1)
        self.sample_per_step = QSpinBox()
        self.sample_per_step.setValue(15)
        self.sample_per_step.setMinimum(1)
        self.max_steps = QSpinBox()
        self.max_steps.setValue(3)
        self.max_steps.setMinimum(1)
        self.epochs = QSpinBox()
        self.epochs.setValue(200)
        self.epochs.setMinimum(1)
        self.batch_size = QSpinBox()
        self.batch_size.setValue(32)
        self.batch_size.setMinimum(1)
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setValue(0.001)
        self.learning_rate.setRange(1e-6, 1.0)
        self.patience = QSpinBox()
        self.patience.setValue(30)
        self.device = QComboBox()
        self.device.addItems(["cpu", "cuda"])
        train_layout.addRow("Initial Train Size:", self.initial_train)
        train_layout.addRow("Samples per Step:", self.sample_per_step)
        train_layout.addRow("Max Steps:", self.max_steps)
        train_layout.addRow("Epochs per Step:", self.epochs)
        train_layout.addRow("Batch Size:", self.batch_size)
        train_layout.addRow("Learning Rate:", self.learning_rate)
        train_layout.addRow("Patience:", self.patience)
        train_layout.addRow("Device:", self.device)
        train_group.setLayout(train_layout)
        layout.addWidget(train_group)

        huds_group = QGroupBox("HUDS Settings")
        huds_layout = QFormLayout()
        self.repeat_times = QSpinBox()
        self.repeat_times.setValue(10)
        self.repeat_times.setMinimum(2)
        self.topk_ratio = QDoubleSpinBox()
        self.topk_ratio.setValue(0.6)
        self.topk_ratio.setRange(0.1, 1.0)
        self.huds_batch = QSpinBox()
        self.huds_batch.setValue(128)
        self.huds_batch.setMinimum(1)
        huds_layout.addRow("Repeat Times:", self.repeat_times)
        huds_layout.addRow("Top-K Ratio:", self.topk_ratio)
        huds_layout.addRow("HUDS Batch Size:", self.huds_batch)
        huds_group.setLayout(huds_layout)
        layout.addWidget(huds_group)

        pool_group = QGroupBox("Candidate Pool")
        pool_layout = QFormLayout()
        self.total_samples = QSpinBox()
        self.total_samples.setValue(600)
        self.total_samples.setMinimum(10)
        pool_layout.addRow("Total Samples:", self.total_samples)
        pool_group.setLayout(pool_layout)
        layout.addWidget(pool_group)

        model_group = QGroupBox("Model Architecture")
        model_layout = QFormLayout()
        self.hidden_dim = QSpinBox()
        self.hidden_dim.setValue(64)
        self.hidden_dim.setMinimum(16)
        self.encoder_blocks = QSpinBox()
        self.encoder_blocks.setValue(2)
        self.encoder_blocks.setMinimum(1)
        self.dropout = QDoubleSpinBox()
        self.dropout.setValue(0.1)
        self.dropout.setRange(0.0, 0.9)
        model_layout.addRow("Hidden Dim:", self.hidden_dim)
        model_layout.addRow("Encoder Blocks:", self.encoder_blocks)
        model_layout.addRow("Dropout:", self.dropout)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        layout.addStretch()
        scroll.setWidget(content)
        main_layout.addWidget(scroll)

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

    def _auto_fill_outputs(self):
        detected = self.window().property("detected_outputs") or []
        if not detected:
            return
        
        self.output_table.setRowCount(len(detected))
        for i, output in enumerate(detected):
            cb = QCheckBox()
            cb.setChecked(True)
            cb_widget = QWidget()
            cb_layout = QHBoxLayout(cb_widget)
            cb_layout.addWidget(cb)
            cb_layout.setAlignment(cb, Qt.AlignCenter)
            cb_layout.setContentsMargins(0, 0, 0, 0)
            self.output_table.setCellWidget(i, 0, cb_widget)

            name_item = QTableWidgetItem(output.get("name", ""))
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.output_table.setItem(i, 1, name_item)

            type_item = QTableWidgetItem(output.get("type", ""))
            type_item.setFlags(type_item.flags() & ~Qt.ItemIsEditable)
            self.output_table.setItem(i, 2, type_item)

    def get_config(self):
        import time
        project_name = self.project_name.text().strip()
        if not project_name:
            project_name = time.strftime("%Y%m%d_%H%M%S")

        output_names = []
        for i in range(self.output_table.rowCount()):
            cb_widget = self.output_table.cellWidget(i, 0)
            cb = cb_widget.findChild(QCheckBox) if cb_widget else None
            if cb and cb.isChecked():
                name_item = self.output_table.item(i, 1)
                if name_item:
                    output_names.append(name_item.text())
        
        manual_outputs = [x.strip() for x in self.output_names.text().split(",") if x.strip()]
        for o in manual_outputs:
            if o not in output_names:
                output_names.append(o)

        selected_vars = []
        for i in range(self.var_table.rowCount()):
            cb_widget = self.var_table.cellWidget(i, 0)
            cb = cb_widget.findChild(QCheckBox) if cb_widget else None
            if cb and cb.isChecked():
                name_item = self.var_table.item(i, 1)
                min_item = self.var_table.item(i, 3)
                max_item = self.var_table.item(i, 4)
                unit_item = self.var_table.item(i, 5)
                
                from huds_app.utils.aedt_parser import parse_value_with_unit
                min_str = min_item.text() if min_item else ""
                max_str = max_item.text() if max_item else ""
                min_num, _ = parse_value_with_unit(min_str)
                max_num, _ = parse_value_with_unit(max_str)

                if min_num is None or max_num is None:
                    continue

                selected_vars.append({
                    "name": name_item.text() if name_item else "",
                    "min": float(min_num),
                    "max": float(max_num),
                    "sample_points": 60,
                    "unit": unit_item.text() if unit_item else "",
                })

        return {
            "project_name": project_name,
            "random_seed": self.random_seed.value(),
            "aedt_project_path": "",
            "design_name": "",
            "variables": selected_vars,
            "candidate_pool": {
                "total_samples": self.total_samples.value()
            },
            "split": {
                "train_split": 0.8,
                "val_split": 0.1,
                "test_split": 0.1,
            },
            "model": {
                "model_type": "vector_to_vector",
                "output_names": output_names,
                "hidden_dim": self.hidden_dim.value(),
                "encoder_blocks": self.encoder_blocks.value(),
                "dropout": self.dropout.value(),
            },
            "training": {
                "initial_train_size": self.initial_train.value(),
                "sample_per_step": self.sample_per_step.value(),
                "max_steps": self.max_steps.value(),
                "epochs_per_step": self.epochs.value(),
                "batch_size": self.batch_size.value(),
                "learning_rate": self.learning_rate.value(),
                "patience": self.patience.value(),
                "device": self.device.currentText(),
            },
            "huds": {
                "repeat_times": self.repeat_times.value(),
                "topk_ratio": self.topk_ratio.value(),
                "batch_size": self.huds_batch.value(),
                "use_faiss": False,
            },
            "steady_state_pct": self.steady_pct.value(),
        }


class SectionTitle(QWidget):
    def __init__(self, text):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)
        label = QLabel(text)
        label.setStyleSheet("font-size: 22px; font-weight: bold; color: #569cd6;")
        layout.addWidget(label)
