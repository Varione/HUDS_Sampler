import json
import os
import sys
import time
import subprocess

from PyQt5.QtCore import QObject, pyqtSignal

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class HUDSWorkerSignals(QObject):
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    r2_signal = pyqtSignal(float)
    step_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    sweep_progress_signal = pyqtSignal(str, int)  # event_type, data


class HUDSWorker:
    def __init__(self, config_dict, run_dir, aedt_path, design_name):
        self.config_dict = config_dict
        self.run_dir = run_dir
        self.aedt_path = aedt_path
        self.design_name = design_name
        self.abort = False
        self.r2_history = []
        self.signals = HUDSWorkerSignals()

    def _log(self, message):
        self.signals.log_signal.emit(str(message))

    def _progress(self, value):
        self.signals.progress_signal.emit(int(value))

    def run(self):
        try:
            from huds_app.core.config import load_config
            from huds_app.interface.workflow import init_run
            from huds_app.sampling.huds import run_huds_sampling
            from huds_app.data.validation import import_labels as il
            from huds_app.model.train import train_model, TrainingAborted

            config_path = os.path.join(self.run_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.config_dict, f, indent=2, ensure_ascii=False)
            time.sleep(0.5)

            self._log("Initializing run...")
            init_run(config_path, self.run_dir, overwrite=True)
            self._log("Run initialized.")

            cfg = load_config(config_path)
            max_steps = cfg.training.max_steps
            self.r2_history = []

            for step in range(1, max_steps + 1):
                if self.abort:
                    self._log("Aborted by user.")
                    self.signals.finished_signal.emit(False, "Aborted by user")
                    return

                progress = int((step - 1) / max_steps * 100)
                self._progress(progress)
                self.signals.step_signal.emit(f"Step {step}/{max_steps}")
                self._log(f"--- Step {step}: HUDS active sampling ---")

                result = run_huds_sampling(self.run_dir, cfg, step)
                n_selected = len(result["selected_ids"])
                self._log(f"  Selected {n_selected} samples")

                request_path = os.path.join(
                    self.run_dir, "requests", f"train_step_{step:03d}_request.csv"
                )
                self._log(f"  Request CSV generated ({os.path.getsize(request_path)} bytes)")

                if self.abort:
                    self._log("Aborted by user.")
                    self.signals.finished_signal.emit(False, "Aborted by user")
                    return

                force_csv = self._run_maxwell_sweep(
                    request_path, config_path, self.aedt_path, self.design_name, step, self.run_dir
                )

                if self.abort:
                    self._log("Aborted by user.")
                    self.signals.finished_signal.emit(False, "Aborted by user")
                    return

                label_path, out_names, skipped_ids = self._extract_labels(
                    force_csv, step, self.run_dir, cfg
                )

                if hasattr(cfg, "model") and hasattr(cfg.model, "output_names"):
                    old_out = cfg.model.output_names
                    cfg.model.output_names = out_names
                    if old_out != out_names:
                        self._log(f"  Updated output_names: {old_out} -> {out_names}")
                        with open(config_path, "r", encoding="utf-8") as f:
                            full_config = json.load(f)
                        full_config["model"]["output_names"] = out_names
                        with open(config_path, "w", encoding="utf-8") as f:
                            json.dump(full_config, f, indent=2, ensure_ascii=False)
                        cfg = load_config(config_path)

                imported = il(self.run_dir, step, label_path, overwrite=False, skipped_ids=skipped_ids)
                self._log(f"  Imported {imported} labels")

                # Allow the training-start dataset summary (reported at 0%) through.
                last_print_pct = [-10]

                def progress_cb(msg, percent):
                    if percent - last_print_pct[0] >= 10 or percent == 100:
                        self._log(f"  [Train] {msg}")
                        last_print_pct[0] = percent

                metrics = train_model(self.run_dir, cfg, progress_cb=progress_cb, cancel_cb=lambda: self.abort)
                r2 = metrics.get("val_r2_avg", float("nan"))
                self.r2_history.append(r2)
                self.signals.r2_signal.emit(r2)
                self._log(f"  Training complete: val_r2_avg = {r2}")

            self._progress(100)
            self.signals.finished_signal.emit(True, f"Completed all {max_steps} steps")
        except TrainingAborted:
            self._log("Workflow aborted by user during training")
            self.signals.finished_signal.emit(False, "Aborted by user")
        except Exception as e:
            self._log(f"Error: {e}")
            import traceback
            self._log(traceback.format_exc())
            self.signals.finished_signal.emit(False, str(e))

    def _run_maxwell_sweep(self, request_csv, cfg_path, aedt_path, dsgn_name, step, run_dir):
        self._log("  Running Maxwell simulation...")
        env = os.environ.copy()
        env["ANSYSLMD_LICENSE_FILE"] = "24500@licensing.hkust.edu.cn"
        env["PYTHONPATH"] = PROJECT_ROOT + ";" + env.get("PYTHONPATH", "")
        sweep_script = os.path.join(
            PROJECT_ROOT, "huds_app", "interface", "maxwell_sweep.py"
        )
        project_dir = os.path.dirname(aedt_path) if os.path.isfile(aedt_path) else aedt_path

        # Progress file for polling sweep progress from subprocess
        progress_file = os.path.join(run_dir, f"sweep_progress_step{step}.json")
        if os.path.exists(progress_file):
            os.remove(progress_file)

        step_output_dir = os.path.join(project_dir, f"sweep_step{step}")
        os.makedirs(step_output_dir, exist_ok=True)

        log_file = os.path.join(run_dir, f"sweep_log_step{step}.txt")
        with open(log_file, "w", encoding="utf-8") as lf:
            proc = subprocess.Popen(
                [sys.executable, sweep_script, "--csv", request_csv,
                 "--config", cfg_path, "--project", aedt_path, "--design", dsgn_name,
                 "--output-dir", step_output_dir, "--progress-file", progress_file],
                stdout=lf, stderr=lf,
                env=env,
            )

        last_progress_reported = ""
        while proc.poll() is None:
            if self.abort:
                self._terminate_process_tree(proc)
                raise RuntimeError("Simulation aborted by user")

            # Poll progress file for lifecycle events
            try:
                if os.path.exists(progress_file):
                    with open(progress_file, "r", encoding="utf-8") as pf:
                        data = json.load(pf)
                    status = data.get("status", "")
                    if status != last_progress_reported:
                        last_progress_reported = status
                        if status == "running":
                            self.signals.sweep_progress_signal.emit("started", 0)
                            self._log("  Sweep running...")
                        elif status == "completed":
                            self.signals.sweep_progress_signal.emit("completed", 1)
                            self._log("  Sweep completed")
                        elif status == "failed":
                            error = data.get("error", "unknown")
                            self.signals.sweep_progress_signal.emit("failed", 0)
                            self._log(f"  Sweep failed: {error}")
            except Exception:
                pass

            time.sleep(0.5)

        proc.wait()
        if proc.returncode != 0:
            try:
                with open(log_file, "r", encoding="utf-8") as lf:
                    log_content = lf.read()[-3000:]
                self._log(f"  Simulation failed (see {log_file}):\n{log_content}")
            except Exception:
                self._log(f"  Simulation failed (log: {log_file})")
            raise RuntimeError("maxwell_sweep.py failed")

        step_output_dir = os.path.join(project_dir, f"sweep_step{step}")
        os.makedirs(step_output_dir, exist_ok=True)

        output_csv = None
        for fname in os.listdir(step_output_dir):
            if fname.endswith(".csv") and fname != "ParametricSetup1_Table.csv":
                fpath = os.path.join(step_output_dir, fname)
                if os.path.getsize(fpath) > 0:
                    output_csv = fpath
                    break

        if not output_csv:
            raise RuntimeError("Simulation output file not found")

        size = os.path.getsize(output_csv)
        self._log(f"  Exported: {output_csv} ({size} bytes)")

        import shutil
        raw_dir = os.path.join(run_dir, "raw_outputs", f"step_{step:03d}")
        os.makedirs(raw_dir, exist_ok=True)
        dest = os.path.join(raw_dir, "force_timeseries.csv")
        shutil.copy2(output_csv, dest)
        self._log(f"  Saved raw timeseries: {dest}")
        return output_csv

    def _terminate_process_tree(self, proc):
        try:
            import subprocess as sp
            result = sp.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, timeout=10,
                creationflags=sp.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass

    def _extract_labels(self, force_csv, step, run_dir, cfg):
        import pandas as pd
        import re
        from huds_app.core.storage import read_csv, write_csv
        from huds_app.data.schema import SAMPLE_ID_COLUMN

        request_path = os.path.join(
            run_dir, "requests", f"train_step_{step:03d}_request.csv"
        )
        df = pd.read_csv(force_csv)
        request_df = read_csv(request_path)

        var_names = [v.name for v in cfg.variables] if cfg.variables else []
        steady_pct = getattr(cfg, "steady_state_pct", 0.2)

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
        self._log(f"  Detected outputs: {out_names}")

        # Check time series length consistency: only keep samples whose effective data length
        # matches the maximum length across all samples. Skip shorter (incomplete) ones.
        col_valid_lengths = {}
        for col, (clean_prefix, vars_in_col) in col_var_map.items():
            valid_count = int(pd.to_numeric(df[col], errors="coerce").notna().sum())
            col_valid_lengths[col] = valid_count
        max_length = max(col_valid_lengths.values()) if col_valid_lengths else 0
        valid_cols = {col for col, length in col_valid_lengths.items() if length == max_length}
        invalid_cols = {col for col, length in col_valid_lengths.items() if length < max_length}

        if invalid_cols:
            self._log(
                f"  Warning: {len(invalid_cols)} sample(s) have incomplete time series data "
                f"(length < {max_length}). These will be skipped and remain in the candidate pool."
            )

        skipped_ids = set()
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

            # Find matching columns for this request row
            matched_col = None
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
                if match:
                    matched_col = col
                    break

            # Skip samples with incomplete time series data
            if matched_col and matched_col not in valid_cols:
                skipped_ids.add(req_row[SAMPLE_ID_COLUMN])
                continue

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

        if not label_rows:
            raise RuntimeError("All samples have incomplete time series data. No valid data available.")

        merged = pd.DataFrame(label_rows)
        cols = [SAMPLE_ID_COLUMN] + var_names + out_names
        label_path = os.path.join(run_dir, "data", f"step{step}_labels.csv")
        write_csv(merged[cols], label_path)
        return label_path, out_names, skipped_ids
