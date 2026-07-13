from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
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

        # Variable names
        add_label("Variable Names (comma separated)")
        self.var_names_edit = QLineEdit("v")
        add_widget(self.var_names_edit, stretch=2)

        # Variable mins
        add_label("Variable Mins (comma separated)")
        self.var_mins_edit = QLineEdit("100")
        add_widget(self.var_mins_edit, stretch=2)

        # Variable maxs
        add_label("Variable Maxs (comma separated)")
        self.var_maxs_edit = QLineEdit("500")
        add_widget(self.var_maxs_edit, stretch=2)

        # Variable units
        add_label("Variable Units (comma separated)")
        self.var_units_edit = QLineEdit("km_per_hour")
        add_widget(self.var_units_edit, stretch=2)

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

    def _parse_list(self, text):
        return [x.strip() for x in text.split(",") if x.strip()]

    def _parse_floats(self, text):
        return [float(x.strip()) for x in text.split(",") if x.strip()]

    def get_config(self):
        var_names = self._parse_list(self.var_names_edit.text())
        var_mins = self._parse_floats(self.var_mins_edit.text())
        var_maxs = self._parse_floats(self.var_maxs_edit.text())
        var_units = self._parse_list(self.var_units_edit.text())

        variables = []
        for i, name in enumerate(var_names):
            variables.append({
                "name": name,
                "min": var_mins[i] if i < len(var_mins) else 0.0,
                "max": var_maxs[i] if i < len(var_maxs) else 1.0,
                "sample_points": 60,
                "unit": var_units[i] if i < len(var_units) else "",
            })

        return {
            "project_name": "",
            "random_seed": 42,
            "aedt_project_path": self.aedt_path_edit.text().strip(),
            "design_name": "",
            "variables": variables,
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
                "output_names": self._parse_list(self.output_names_edit.text()),
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
