import json
import os
import sys
import time

import numpy as np
import pandas as pd

from PyQt5.QtCore import QThread, pyqtSignal

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from huds_app.core.config import load_config
from huds_app.core.storage import read_csv, write_csv
from huds_app.data.schema import SAMPLE_ID_COLUMN


class HUDSWorker(QThread):
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    r2_signal = pyqtSignal(float)
    step_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, config_dict, run_dir, aedt_path, design_name):
        super().__init__()
        self.config_dict = config_dict
        self.run_dir = run_dir
        self.aedt_path = aedt_path
        self.design_name = design_name
        self.abort = False

    def log(self, message):
        self.log_signal.emit(str(message))

    def run(self):
        from huds_app.sampling.huds import run_huds_sampling
        from huds_app.data.validation import import_labels as il
        from huds_app.model.train import train_model
        from huds_app.interface.workflow import show_status

        config_path = os.path.join(self.run_dir, "config.json")
        cfg = load_config(config_path)
        max_steps = cfg.training.max_steps

        r2_history = []

        for step in range(1, max_steps + 1):
            if self.abort:
                self.finished_signal.emit(False, "训练已被用户中止")
                return

            self.step_signal.emit(f"步骤 {step}/{max_steps}: HUDS 主动采样")
            progress_base = int((step - 1) / max_steps * 100)
            self.progress_signal.emit(progress_base)

            try:
                result = run_huds_sampling(self.run_dir, cfg, step)
                n_selected = len(result["selected_ids"])
                self.log(f"  选择了 {n_selected} 个样本")

            except Exception as e:
                self.log(f"  HUDS 采样失败: {e}")
                self.finished_signal.emit(False, f"步骤 {step} HUDS 采样失败: {e}")
                return

            request_path = os.path.join(
                self.run_dir, "requests", f"train_step_{step:03d}_request.csv"
            )
            try:
                req_df = pd.read_csv(request_path, dtype={"sample_id": str})
                self.log(f"  请求表已生成 ({len(req_df)} 行)")
            except Exception as e:
                self.log(f"  读取请求表失败: {e}")
                self.finished_signal.emit(False, f"步骤 {step} 读取请求表失败")
                return

            try:
                force_csv = self._run_maxwell_sweep(request_path, config_path, step)
            except Exception as e:
                self.log(f"  Maxwell 仿真失败: {e}")
                self.finished_signal.emit(False, f"步骤 {step} 仿真失败: {e}")
                return

            try:
                label_path, out_names = self._extract_labels(force_csv, step, cfg)
            except Exception as e:
                self.log(f"  提取标签失败: {e}")
                self.finished_signal.emit(False, f"步骤 {step} 提取标签失败: {e}")
                return

            if hasattr(cfg, "model") and hasattr(cfg.model, "output_names"):
                old_out = cfg.model.output_names
                cfg.model.output_names = out_names
                if old_out != out_names:
                    self.log(f"  更新 output_names: {old_out} -> {out_names}")
                    with open(config_path, "r", encoding="utf-8") as f:
                        full_config = json.load(f)
                    full_config["model"]["output_names"] = out_names
                    with open(config_path, "w", encoding="utf-8") as f:
                        json.dump(full_config, f, indent=2, ensure_ascii=False)
                    cfg = load_config(config_path)

            try:
                imported = il(self.run_dir, step, label_path, overwrite=True)
                self.log(f"  导入了 {imported} 条标签")
            except Exception as e:
                self.log(f"  导入标签失败: {e}")
                self.finished_signal.emit(False, f"步骤 {step} 导入标签失败: {e}")
                return

            last_print_pct = [0]

            def progress_cb(msg, percent):
                if percent - last_print_pct[0] >= 10 or percent == 100:
                    self.log(f"  [Train] {msg}")
                    train_progress = int(
                        progress_base + (step - 1) / max_steps * 100 + percent / 10
                    )
                    self.progress_signal.emit(min(99, train_progress))
                    last_print_pct[0] = percent

            try:
                metrics = train_model(self.run_dir, cfg, progress_cb=progress_cb)
            except Exception as e:
                self.log(f"  训练失败: {e}")
                self.finished_signal.emit(False, f"步骤 {step} 训练失败: {e}")
                return

            r2 = metrics.get("val_r2_avg", float("nan"))
            r2_history.append(r2)
            self.r2_signal.emit(r2)
            r2_str = f"{r2:.4f}" if not np.isnan(r2) else "N/A (样本不足)"
            self.log(f"  训练完成: val_r2_avg = {r2_str}")

            try:
                show_status(self.run_dir)
            except Exception:
                pass

            self.progress_signal.emit(int(step / max_steps * 100))

        self.progress_signal.emit(100)
        valid_r2 = [r for r in r2_history if not np.isnan(r)]
        if len(valid_r2) >= 2:
            summary = f"R2 变化: {valid_r2[0]:.4f} -> {valid_r2[-1]:.4f}"
        elif valid_r2:
            summary = f"最终 R2: {valid_r2[0]:.4f}"
        else:
            summary = "R2 不可用 (验证集样本不足)"

        self.finished_signal.emit(True, summary)

    def _run_maxwell_sweep(self, request_csv, cfg_path, step):
        self.log("  运行 Maxwell 仿真...")
        env = os.environ.copy()
        env["ANSYSLMD_LICENSE_FILE"] = "24500@licensing.hkust.edu.cn"
        env["PYTHONPATH"] = PROJECT_ROOT + ";" + env.get("PYTHONPATH", "")
        sweep_script = os.path.join(
            PROJECT_ROOT, "huds_app", "interface", "maxwell_sweep.py"
        )

        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                sweep_script,
                "--csv",
                request_csv,
                "--config",
                cfg_path,
                "--project",
                self.aedt_path,
                "--design",
                self.design_name,
                "--output-dir",
                os.path.dirname(self.aedt_path) if os.path.isfile(self.aedt_path) else self.aedt_path,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            env=env,
        )

        if result.returncode != 0:
            err = result.stderr.replace("\ufffd", "?")[:2000]
            self.log(f"  仿真失败:\n{err}")
            out = result.stdout.replace("\ufffd", "?")[:1000]
            if out:
                self.log(f"  输出:\n{out}")
            raise RuntimeError("maxwell_sweep.py failed")

        project_dir = os.path.dirname(self.aedt_path) if os.path.isfile(self.aedt_path) else self.aedt_path
        output_csv = os.path.join(project_dir, "Force Plot 1.csv")
        if os.path.exists(output_csv):
            size = os.path.getsize(output_csv)
            self.log(f"  已导出: {output_csv} ({size} bytes)")

            import shutil
            raw_dir = os.path.join(self.run_dir, "raw_outputs")
            os.makedirs(raw_dir, exist_ok=True)
            dest = os.path.join(raw_dir, f"step{step}_force_timeseries.csv")
            shutil.copy2(output_csv, dest)
            self.log(f"  已保存原始时序文件: {dest}")
            return output_csv
        raise RuntimeError("仿真结果文件不存在")

    def _extract_labels(self, force_csv, step, cfg):
        request_path = os.path.join(
            self.run_dir, "requests", f"train_step_{step:03d}_request.csv"
        )
        df = pd.read_csv(force_csv)
        request_df = read_csv(request_path)

        var_name = cfg.variables[0].name if cfg.variables else "v"
        steady_pct = getattr(cfg, "steady_state_pct", 0.2)

        detected_outputs = set()
        for col in df.columns[1:]:
            if f"{var_name}='" not in col:
                continue
            prefix = col.rsplit(f" - {var_name}='", 1)[0]
            detected_outputs.add(prefix)

        out_names = sorted(detected_outputs) if detected_outputs else ["peak_force_y", "peak_force_z"]
        self.log(f"  检测到输出: {out_names}")
        self.log(f"  稳态值提取: 末尾 {steady_pct * 100:.0f}% 数据均值")

        force_data = []
        for col in df.columns[1:]:
            if f"{var_name}='" not in col:
                continue
            v_str = col.split(f"{var_name}='")[1].split("'")[0]
            v_val = float(v_str.replace("km_per_hour", ""))

            n_total = len(df[col])
            n_steady = max(1, int(n_total * steady_pct))
            steady_mean = df[col].iloc[-n_steady:].mean()

            prefix = col.rsplit(f" - {var_name}='", 1)[0]
            force_data.append({
                "v_rounded": round(v_val, 4),
                "value": steady_mean,
                "output_name": prefix,
            })

        if not force_data:
            raise RuntimeError("未能从仿真结果中提取任何数据")

        force_df = pd.DataFrame(force_data)
        pivoted = force_df.pivot(
            index="v_rounded", columns="output_name", values="value"
        ).reset_index()

        merged = request_df.copy()
        merged["v_rounded"] = merged[var_name].round(4)
        merged = merged.merge(pivoted, on="v_rounded", how="left")
        merged = merged.drop(columns=["v_rounded"])

        label_path = os.path.join(self.run_dir, "data", f"step{step}_labels.csv")
        write_csv(merged[[SAMPLE_ID_COLUMN, var_name] + out_names], label_path)
        return label_path, out_names
