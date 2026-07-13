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
    QGroupBox,
    QPushButton,
    QListWidget,
    QListWidgetItem,
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

        self.var_list = QListWidget()
        var_layout.addWidget(self.var_list)

        manual_group = QGroupBox("手动覆盖 (可选)")
        manual_layout = QVBoxLayout()

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("变量名:"))
        self.var_names_edit = QLineEdit("")
        row1.addWidget(self.var_names_edit, 1)
        manual_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("最小值:"))
        self.var_mins_edit = QLineEdit("")
        row2.addWidget(self.var_mins_edit, 1)
        manual_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("最大值:"))
        self.var_maxs_edit = QLineEdit("")
        row3.addWidget(self.var_maxs_edit, 1)
        manual_layout.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("单位:"))
        self.var_units_edit = QLineEdit("")
        row4.addWidget(self.var_units_edit, 1)
        manual_layout.addLayout(row4)

        manual_group.setLayout(manual_layout)
        layout.addWidget(manual_group)

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

    def _auto_fill_variables(self):
        detected = self.window().property("detected_variables") or []
        if not detected:
            return
        
        from huds_app.utils.aedt_parser import parse_value_with_unit
        self.var_list.clear()
        for v in detected:
            num_val, unit = parse_value_with_unit(v.get('default', ''))
            min_val = v.get('min', '')
            max_val = v.get('max', '')
            min_num, _ = parse_value_with_unit(min_val)
            max_num, _ = parse_value_with_unit(max_val)
            
            display = f"{v['name']} = {v.get('default', 'N/A')}"
            if min_val and max_val:
                display += f"  [{min_val} ~ {max_val}]"
            item = QListWidgetItem(display)
            item.setData(1, v)
            self.var_list.addItem(item)
    
    def initializePage(self):
        detected = self.window().property("detected_variables") or []
        if detected:
            self._auto_fill_variables()
    
    def validatePage(self):
        from huds_app.utils.aedt_parser import parse_value_with_unit
        
        # Collect variables from list (auto-detected) and manual override
        variables = []
        
        # First, add auto-detected variables that have min/max
        for i in range(self.var_list.count()):
            item = self.var_list.item(i)
            v = item.data(1)
            if not v:
                continue
            
            num_val, unit = parse_value_with_unit(v.get('default', ''))
            min_num, _ = parse_value_with_unit(v.get('min', ''))
            max_num, _ = parse_value_with_unit(v.get('max', ''))
            
            # Only include variables that have optimization bounds
            if min_num is not None and max_num is not None:
                variables.append({
                    "name": v['name'],
                    "min": float(min_num),
                    "max": float(max_num),
                    "sample_points": 60,
                    "unit": unit,
                })
        
        # Then add manually specified variables
        var_names = [x.strip() for x in self.var_names_edit.text().split(",") if x.strip()]
        var_mins = [float(x.strip()) for x in self.var_mins_edit.text().split(",") if x.strip()]
        var_maxs = [float(x.strip()) for x in self.var_maxs_edit.text().split(",") if x.strip()]
        var_units_raw = self.var_units_edit.text()
        var_units = [x.strip() for x in var_units_raw.split(",") if x.strip()]
        
        while len(var_units) < len(var_names):
            var_units.append("")
        
        existing_names = {v['name'] for v in variables}
        for i, name in enumerate(var_names):
            if name not in existing_names:
                variables.append({
                    "name": name,
                    "min": var_mins[i] if i < len(var_mins) else 0.0,
                    "max": var_maxs[i] if i < len(var_maxs) else 1.0,
                    "sample_points": 60,
                    "unit": var_units[i] if i < len(var_units) else "",
                })

        output_names = [x.strip() for x in self.output_names_edit.text().split(",") if x.strip()]

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
