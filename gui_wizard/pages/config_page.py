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
    QFileDialog,
    QGroupBox,
)


class ConfigPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("配置参数")
        self.setSubTitle("设置仿真变量、训练参数和计算设备")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        aedt_group = QGroupBox("AEDT 项目路径")
        aedt_layout = QHBoxLayout()
        self.aedt_path_edit = QLineEdit()
        self.aedt_browse_btn = QPushButton("浏览...")
        self.aedt_browse_btn.clicked.connect(self._browse_aedt)
        aedt_layout.addWidget(QLabel("项目文件:"))
        aedt_layout.addWidget(self.aedt_path_edit, 1)
        aedt_layout.addWidget(self.aedt_browse_btn)
        aedt_group.setLayout(aedt_layout)
        layout.addWidget(aedt_group)

        var_group = QGroupBox("扫参变量")
        var_layout = QVBoxLayout()
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("变量名:"))
        self.var_names_edit = QLineEdit("v")
        row1.addWidget(self.var_names_edit, 1)
        var_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("最小值:"))
        self.var_mins_edit = QLineEdit("100")
        row2.addWidget(self.var_mins_edit, 1)
        var_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("最大值:"))
        self.var_maxs_edit = QLineEdit("500")
        row3.addWidget(self.var_maxs_edit, 1)
        var_layout.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("单位:"))
        self.var_units_edit = QLineEdit("km_per_hour")
        row4.addWidget(self.var_units_edit, 1)
        var_layout.addLayout(row4)

        row5 = QHBoxLayout()
        row5.addWidget(QLabel("输出变量:"))
        self.output_names_edit = QLineEdit("peak_force_y,peak_force_z")
        row5.addWidget(self.output_names_edit, 1)
        var_layout.addLayout(row5)

        row6 = QHBoxLayout()
        row6.addWidget(QLabel("稳态提取比例:"))
        self.steady_pct_spin = QDoubleSpinBox()
        self.steady_pct_spin.setRange(0.01, 1.0)
        self.steady_pct_spin.setSingleStep(0.05)
        self.steady_pct_spin.setValue(0.2)
        row6.addWidget(self.steady_pct_spin)
        var_layout.addLayout(row6)

        var_group.setLayout(var_layout)
        layout.addWidget(var_group)

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

    def _browse_aedt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 AEDT 项目文件", "", "AEDT Files (*.aedt)"
        )
        if path:
            self.aedt_path_edit.setText(path)

    def initializePage(self):
        pass

    def validatePage(self):
        var_names = [x.strip() for x in self.var_names_edit.text().split(",") if x.strip()]
        var_mins = [float(x.strip()) for x in self.var_mins_edit.text().split(",") if x.strip()]
        var_maxs = [float(x.strip()) for x in self.var_maxs_edit.text().split(",") if x.strip()]
        var_units_raw = self.var_units_edit.text()
        var_units = [x.strip() for x in var_units_raw.split(",") if x.strip()]
        output_names = [x.strip() for x in self.output_names_edit.text().split(",") if x.strip()]

        while len(var_units) < len(var_names):
            var_units.append("")

        variables = []
        for i, name in enumerate(var_names):
            variables.append({
                "name": name,
                "min": var_mins[i] if i < len(var_mins) else 0.0,
                "max": var_maxs[i] if i < len(var_maxs) else 1.0,
                "sample_points": 60,
                "unit": var_units[i] if i < len(var_units) else "",
            })

        config = {
            "project_name": time.strftime("%Y%m%d_%H%M%S"),
            "random_seed": 42,
            "aedt_project_path": self.aedt_path_edit.text().strip(),
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
