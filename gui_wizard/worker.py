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
        from huds_app.interface.workflow import init_run, show_status

        config_path = os.path.join(self.run_dir, "config.json")

        self.log("Initializing run...")
        result = init_run(config_path, self.run_dir, overwrite=True)
        self.log(f"Run initialized, candidate pool size: {result['total_candidates']}")

        try:
            show_status(self.run_dir)
        except Exception:
            pass

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
        os.environ["ANSYSLMD_LICENSE_FILE"] = "24500@licensing.hkust.edu.cn"

        project_dir = os.path.dirname(self.aedt_path) if os.path.isfile(self.aedt_path) else self.aedt_path

        from huds_app.interface.maxwell_sweep import run_sweep
        exported = run_sweep(
            csv_path=request_csv,
            config_path=cfg_path,
            project_path=self.aedt_path,
            design_name=self.design_name,
            output_dir=project_dir,
        )

        self.log(f"  仿真完成，导出了 {len(exported)} 个文件")

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

        var_names = [v.name for v in cfg.variables] if cfg.variables else []
        steady_pct = getattr(cfg, "steady_state_pct", 0.2)

        import re
        kv_pattern = re.compile(r"(\w+)='([^']+)'")

        def parse_col_vars(col_name):
            matches = kv_pattern.findall(col_name)
            result = {}
            for k, v in matches:
                clean = v.replace("km_per_hour", "").replace("Hz", "").replace("A", "")
                try:
                    result[k] = float(clean)
                except ValueError:
                    result[k] = clean
            return result

        detected_outputs = set()
        col_var_map = {}
        for col in df.columns[1:]:
            vars_in_col = parse_col_vars(col)
            if not vars_in_col:
                continue
            prefix = col.rsplit(" - ", 1)[0] if " - " in col else col.split("[")[0].strip()
            clean_prefix = re.sub(r'\s*\[.*?\]', '', prefix).strip()
            detected_outputs.add(clean_prefix)
            col_var_map[col] = (clean_prefix, vars_in_col)

        out_names = sorted(detected_outputs) if detected_outputs else ["peak_force_y", "peak_force_z"]
        self.log(f"  检测到输出: {out_names}")
        self.log(f"  稳态值提取: 末尾 {steady_pct * 100:.0f}% 数据均值")

        self.log(f"  请求表变量: {var_names}")
        self.log(f"  请求表行数: {len(request_df)}")
        self.log(f"  仿真结果列数: {len(df.columns)-1}")

        label_rows = []
        for _, req_row in request_df.iterrows():
            req_vals = {}
            for vn in var_names:
                if vn in req_row.index:
                    raw = str(req_row[vn]).replace("km_per_hour", "").replace("Hz", "").replace("A", "")
                    try:
                        req_vals[vn] = float(raw)
                    except ValueError:
                        req_vals[vn] = raw

            row_data = {SAMPLE_ID_COLUMN: req_row[SAMPLE_ID_COLUMN]}
            for vn in var_names:
                if vn in req_row.index:
                    row_data[vn] = req_row[vn]

            matched_outputs = {}
            for col, (clean_prefix, vars_in_col) in col_var_map.items():
                match = True
                for vn in var_names:
                    if vn not in vars_in_col or vn not in req_vals:
                        match = False
                        break
                    if isinstance(vars_in_col[vn], float) and isinstance(req_vals[vn], float):
                        if abs(vars_in_col[vn] - req_vals[vn]) > 1e-6:
                            match = False
                    else:
                        if str(vars_in_col[vn]) != str(req_vals[vn]):
                            match = False

                if not match:
                    continue

                n_total = len(df[col])
                n_steady = max(1, int(n_total * steady_pct))
                steady_mean = df[col].iloc[-n_steady:].mean()
                matched_outputs[clean_prefix] = steady_mean

            row_data.update(matched_outputs)
            label_rows.append(row_data)

        self.log(f"  匹配到 {len(label_rows)} 行, 每行 {len(label_rows[0]) if label_rows else 0} 列")

        if not label_rows:
            raise RuntimeError("未能从仿真结果中提取任何数据")

        has_outputs = any(any(k in out_names and v is not None for k, v in r.items()) for r in label_rows)
        if not has_outputs:
            raise RuntimeError("仿真结果与请求表变量不匹配，请检查变量名和单位")

        merged = pd.DataFrame(label_rows)

        label_path = os.path.join(self.run_dir, "data", f"step{step}_labels.csv")
        cols = [SAMPLE_ID_COLUMN] + var_names + out_names
        write_csv(merged[cols], label_path)
        return label_path, out_names
