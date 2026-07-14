import time
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox,
    QWizardPage,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QDoubleSpinBox,
    QSpinBox,
    QComboBox,
    QGroupBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QWidget,
    QMessageBox,
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

        var_group = QGroupBox("扫参变量")
        var_layout = QVBoxLayout()

        auto_row = QHBoxLayout()
        auto_row.addWidget(QLabel("已检测到的设计变量:"))
        self.auto_detect_btn = QPushButton("自动填充")
        self.auto_detect_btn.clicked.connect(self._auto_fill_variables)
        auto_row.addWidget(self.auto_detect_btn)
        var_layout.addLayout(auto_row)

        self.var_table = QTableWidget(0, 6)
        self.var_table.setHorizontalHeaderLabels(["选择", "名称", "默认值", "最小值", "最大值", "单位"])
        self.var_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        var_layout.addWidget(self.var_table)

        var_group.setLayout(var_layout)
        layout.addWidget(var_group)

        output_group = QGroupBox("输出变量设置")
        output_layout = QVBoxLayout()

        auto_out_row = QHBoxLayout()
        auto_out_row.addWidget(QLabel("已检测到的输出变量:"))
        self.auto_detect_outputs_btn = QPushButton("自动填充")
        self.auto_detect_outputs_btn.clicked.connect(self._auto_fill_outputs)
        auto_out_row.addWidget(self.auto_detect_outputs_btn)
        output_layout.addLayout(auto_out_row)

        self.output_table = QTableWidget(0, 3)
        self.output_table.setHorizontalHeaderLabels(["选择", "名称", "类型"])
        self.output_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.output_table.setColumnHidden(2, False)
        output_layout.addWidget(self.output_table)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("手动添加 (逗号分隔):"))
        self.output_names_edit = QLineEdit("")
        row2.addWidget(self.output_names_edit, 1)
        output_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("稳态提取比例:"))
        self.steady_pct_spin = QDoubleSpinBox()
        self.steady_pct_spin.setRange(0.01, 1.0)
        self.steady_pct_spin.setSingleStep(0.05)
        self.steady_pct_spin.setValue(0.2)
        row3.addWidget(self.steady_pct_spin)
        output_layout.addLayout(row3)

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

    def _auto_fill_outputs(self):
        wizard = self.window()
        detected = wizard.property("detected_outputs") or []
        if not detected:
            QMessageBox.information(self, "调试", f"未检测到输出变量。\nWizard属性: {list(wizard.dynamicPropertyNames())}")
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

    def _auto_fill_variables(self):
        wizard = self.window()
        detected = wizard.property("detected_variables") or []
        if not detected:
            QMessageBox.information(self, "调试", f"未检测到设计变量。\nWizard属性: {list(wizard.dynamicPropertyNames())}")
            return
        
        self.var_table.setRowCount(len(detected))
        for i, var in enumerate(detected):
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
    
    def initializePage(self):
        wizard = self.window()
        if not wizard:
            print("[ConfigPage] No wizard window found")
            return
        
        detected_vars = wizard.property("detected_variables") or []
        print(f"[ConfigPage] Detected variables: {len(detected_vars)}")
        if detected_vars:
            self._auto_fill_variables()
        
        detected_outputs = wizard.property("detected_outputs") or []
        print(f"[ConfigPage] Detected outputs: {len(detected_outputs)}")
        if detected_outputs:
            self._auto_fill_outputs()
    
    def validatePage(self):
        from huds_app.utils.aedt_parser import parse_value_with_unit
        
        variables = []
        for i in range(self.var_table.rowCount()):
            cb_widget = self.var_table.cellWidget(i, 0)
            cb = cb_widget.findChild(QCheckBox) if cb_widget else None
            if cb and cb.isChecked():
                name_item = self.var_table.item(i, 1)
                min_item = self.var_table.item(i, 3)
                max_item = self.var_table.item(i, 4)
                unit_item = self.var_table.item(i, 5)

                min_str = min_item.text() if min_item else ""
                max_str = max_item.text() if max_item else ""
                min_num, _ = parse_value_with_unit(min_str)
                max_num, _ = parse_value_with_unit(max_str)

                if min_num is not None and max_num is not None:
                    variables.append({
                        "name": name_item.text() if name_item else "",
                        "min": float(min_num),
                        "max": float(max_num),
                        "sample_points": 60,
                        "unit": unit_item.text() if unit_item else "",
                    })

        if not variables:
            QMessageBox.warning(self, "警告", "请至少选择一个变量并填写最小值和最大值")
            return False

        # Collect output names from table and manual input
        output_names = []
        for i in range(self.output_table.rowCount()):
            cb_widget = self.output_table.cellWidget(i, 0)
            cb = cb_widget.findChild(QCheckBox) if cb_widget else None
            if cb and cb.isChecked():
                name_item = self.output_table.item(i, 1)
                if name_item:
                    output_names.append(name_item.text())
        
        # Add manually specified outputs
        manual_outputs = [x.strip() for x in self.output_names_edit.text().split(",") if x.strip()]
        for o in manual_outputs:
            if o not in output_names:
                output_names.append(o)

        config = {
            "project_name": time.strftime("%Y%m%d_%H%M%S"),
            "random_seed": 42,
            "aedt_project_path": self.window().property("aedt_path") or "",
            "variables": variables,
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
