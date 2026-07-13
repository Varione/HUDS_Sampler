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


class HUDSWorker:
    def __init__(self, config_dict, run_dir, aedt_path, design_name):
        self.config_dict = config_dict
        self.run_dir = run_dir
        self.aedt_path = aedt_path
        self.design_name = design_name
        self.abort = False
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
            from huds_app.model.train import train_model

            config_path = os.path.join(self.run_dir, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.config_dict, f, indent=2, ensure_ascii=False)
            time.sleep(0.5)

            self._log("Initializing run...")
            init_run(config_path, self.run_dir)
            self._log("Run initialized.")

            cfg = load_config(config_path)
            max_steps = cfg.training.max_steps
            r2_history = []

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
                    request_path, config_path, self.aedt_path, self.design_name, step, run_dir
                )

                if self.abort:
                    self._log("Aborted by user.")
                    self.signals.finished_signal.emit(False, "Aborted by user")
                    return

                label_path, out_names = self._extract_labels(
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

                imported = il(self.run_dir, step, label_path, overwrite=True)
                self._log(f"  Imported {imported} labels")

                last_print_pct = [0]

                def progress_cb(msg, percent):
                    if percent - last_print_pct[0] >= 10 or percent == 100:
                        self._log(f"  [Train] {msg}")
                        last_print_pct[0] = percent

                metrics = train_model(self.run_dir, cfg, progress_cb=progress_cb)
                r2 = metrics.get("val_r2_avg", float("nan"))
                r2_history.append(r2)
                self.signals.r2_signal.emit(r2)
                self._log(f"  Training complete: val_r2_avg = {r2}")

            self._progress(100)
            self.signals.finished_signal.emit(True, f"Completed all {max_steps} steps")
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
        result = subprocess.run(
            [sys.executable, sweep_script, "--csv", request_csv,
             "--config", cfg_path, "--project", aedt_path, "--design", dsgn_name,
             "--output-dir", os.path.dirname(aedt_path) if os.path.isfile(aedt_path) else aedt_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=600, env=env,
        )
        if result.returncode != 0:
            err = result.stderr.replace("\ufffd", "?")[:2000]
            self._log(f"  Simulation failed:\n{err}")
            out = result.stdout.replace("\ufffd", "?")[:1000]
            if out:
                self._log(f"  Output:\n{out}")
            raise RuntimeError("maxwell_sweep.py failed")

        project_dir = os.path.dirname(aedt_path) if os.path.isfile(aedt_path) else aedt_path
        output_csv = os.path.join(project_dir, "Force Plot 1.csv")
        if os.path.exists(output_csv):
            size = os.path.getsize(output_csv)
            self._log(f"  Exported: {output_csv} ({size} bytes)")

            import shutil
            raw_dir = os.path.join(run_dir, "raw_outputs")
            os.makedirs(raw_dir, exist_ok=True)
            dest = os.path.join(raw_dir, f"step{step}_force_timeseries.csv")
            shutil.copy2(output_csv, dest)
            self._log(f"  Saved raw timeseries: {dest}")
            return output_csv
        raise RuntimeError("Simulation output file not found")

    def _extract_labels(self, force_csv, step, run_dir, cfg):
        import pandas as pd
        from huds_app.core.storage import read_csv, write_csv
        from huds_app.data.schema import SAMPLE_ID_COLUMN

        request_path = os.path.join(
            run_dir, "requests", f"train_step_{step:03d}_request.csv"
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
        self._log(f"  Detected outputs: {out_names}")

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
            raise RuntimeError("Could not extract data from simulation output")

        force_df = pd.DataFrame(force_data)
        pivoted = force_df.pivot(index="v_rounded", columns="output_name", values="value").reset_index()

        merged = request_df.copy()
        merged["v_rounded"] = merged[var_name].round(4)
        merged = merged.merge(pivoted, on="v_rounded", how="left")
        merged = merged.drop(columns=["v_rounded"])

        label_path = os.path.join(run_dir, "data", f"step{step}_labels.csv")
        write_csv(merged[[SAMPLE_ID_COLUMN, var_name] + out_names], label_path)
        return label_path, out_names
