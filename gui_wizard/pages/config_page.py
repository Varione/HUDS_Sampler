import time
from PyQt5.QtWidgets import (
    QWizardPage,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QDoubleSpinBox,
    QSpinBox,
    QComboBox,
    QPushButton,
    QGroupBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QCheckBox,
    QWidget,
)


class ConfigPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("配置参数")
        self.setSubTitle("从 AEDT 设计中选择扫参变量，设置训练参数")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        var_group = QGroupBox("扫参变量 (从设计中选择)")
        var_layout = QVBoxLayout()

        self.var_table = QTableWidget(0, 5)
        self.var_table.setHorizontalHeaderLabels(["选择", "变量名", "默认值", "单位", "说明"])
        self.var_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.var_table.setColumnHidden(4, True)
        var_layout.addWidget(self.var_table)

        refresh_btn = QPushButton("刷新变量列表")
        refresh_btn.clicked.connect(self._refresh_variables)
        var_layout.addWidget(refresh_btn)

        layout.addWidget(QLabel("提示: 如果自动检测失败，请手动输入变量名（逗号分隔）"))
        self.manual_var_edit = QLineEdit()
        self.manual_var_edit.setPlaceholderText("例如: v1, v2, v3")
        var_layout.addWidget(self.manual_var_edit)

        var_group.setLayout(var_layout)
        layout.addWidget(var_group)

        output_group = QGroupBox("输出设置")
        output_layout = QVBoxLayout()

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("仿真输出变量:"))
        self.output_names_edit = QLineEdit("peak_force_y,peak_force_z")
        row1.addWidget(self.output_names_edit, 1)
        output_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("稳态提取比例:"))
        self.steady_pct_spin = QDoubleSpinBox()
        self.steady_pct_spin.setRange(0.01, 1.0)
        self.steady_pct_spin.setSingleStep(0.05)
        self.steady_pct_spin.setValue(0.2)
        row2.addWidget(self.steady_pct_spin)
        output_layout.addLayout(row2)

        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        train_group = QGroupBox("训练设置")
        train_layout = QVBoxLayout()

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("候选池总样本数:"))
        self.total_samples_spin = QSpinBox()
        self.total_samples_spin.setRange(10, 10000)
        self.total_samples_spin.setValue(600)
        row1.addWidget(self.total_samples_spin)
        train_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("初始训练样本数:"))
        self.initial_train_spin = QSpinBox()
        self.initial_train_spin.setRange(5, 500)
        self.initial_train_spin.setValue(20)
        row2.addWidget(self.initial_train_spin)
        train_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("每步新增样本数:"))
        self.sample_per_step_spin = QSpinBox()
        self.sample_per_step_spin.setRange(1, 100)
        self.sample_per_step_spin.setValue(15)
        row3.addWidget(self.sample_per_step_spin)
        train_layout.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("最大迭代轮数:"))
        self.max_steps_spin = QSpinBox()
        self.max_steps_spin.setRange(1, 20)
        self.max_steps_spin.setValue(3)
        row4.addWidget(self.max_steps_spin)
        train_layout.addLayout(row4)

        row5 = QHBoxLayout()
        row5.addWidget(QLabel("每步 Epoch 数:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 2000)
        self.epochs_spin.setValue(200)
        row5.addWidget(self.epochs_spin)
        train_layout.addLayout(row5)

        row6 = QHBoxLayout()
        row6.addWidget(QLabel("计算设备:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda"])
        row6.addWidget(self.device_combo)
        train_layout.addLayout(row6)

        train_group.setLayout(train_layout)
        layout.addWidget(train_group)
        layout.addStretch()

    def _refresh_variables(self):
        wizard = self.window()
        detected = wizard.property("detected_variables") or []
        self._populate_var_table(detected)

    def _populate_var_table(self, detected_vars):
        self.var_table.setRowCount(len(detected_vars))
        for i, var in enumerate(detected_vars):
            cb = QCheckBox()
            cb.setChecked(True)
            cb_widget = QWidget()
            cb_layout = QHBoxLayout(cb_widget)
            cb_layout.addWidget(cb)
            cb_layout.setAlignment(cb, 0x02 | 0x40)
            cb_layout.setContentsMargins(0, 0, 0, 0)
            self.var_table.setCellWidget(i, 0, cb_widget)

            name_item = QTableWidgetItem(var.get("name", ""))
            name_item.setFlags(name_item.flags() & ~2)
            self.var_table.setItem(i, 1, name_item)

            val_item = QTableWidgetItem(var.get("value", ""))
            val_item.setFlags(val_item.flags() & ~2)
            self.var_table.setItem(i, 2, val_item)

            unit_item = QTableWidgetItem(var.get("unit", ""))
            unit_item.setFlags(unit_item.flags() & ~2)
            self.var_table.setItem(i, 3, unit_item)

    def initializePage(self):
        wizard = self.window()
        detected = wizard.property("detected_variables") or []
        self._populate_var_table(detected)

    def validatePage(self):
        selected_vars = []
        for i in range(self.var_table.rowCount()):
            cb_widget = self.var_table.cellWidget(i)
            cb = cb_widget.findChild(QCheckBox) if cb_widget else None
            if cb and cb.isChecked():
                name_item = self.var_table.item(i, 1)
                unit_item = self.var_table.item(i, 3)
                var_name = name_item.text() if name_item else ""
                var_unit = unit_item.text() if unit_item else ""
                selected_vars.append({
                    "name": var_name,
                    "min": 0.0,
                    "max": 1.0,
                    "sample_points": 60,
                    "unit": var_unit,
                })

        # If no variables selected from table, try manual input
        if not selected_vars:
            manual_text = self.manual_var_edit.text().strip()
            if manual_text:
                for vname in [x.strip() for x in manual_text.split(",") if x.strip()]:
                    selected_vars.append({
                        "name": vname,
                        "min": 0.0,
                        "max": 1.0,
                        "sample_points": 60,
                        "unit": "",
                    })

        output_names = [x.strip() for x in self.output_names_edit.text().split(",") if x.strip()]

        config = {
            "project_name": time.strftime("%Y%m%d_%H%M%S"),
            "random_seed": 42,
            "aedt_project_path": self.window().property("aedt_path") or "",
            "variables": selected_vars,
            "candidate_pool": {"total_samples": self.total_samples_spin.value()},
            "split": {"train_split": 0.8, "val_split": 0.1, "test_split": 0.1},
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

        wizard = self.window()
        wizard.setProperty("config", config)
        return True

    def set_next_id(self, nid):
        self._next_id = nid

    def nextId(self):
        return getattr(self, "_next_id", -1)
