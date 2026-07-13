import json
import os
import sys
import time
import subprocess

from PySide6.QtCore import QThread, Signal

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class HUDSWorker(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(int)
    step_done_signal = Signal(int, dict)
    finished_all_signal = Signal(list)
    error_signal = Signal(str)

    def __init__(self, config_dict, aedt_path, design_name, project_dir):
        super().__init__()
        self.config_dict = config_dict
        self.aedt_path = aedt_path
        self.design_name = design_name
        self.project_dir = project_dir
        self._abort = False

    def abort(self):
        self._abort = True

    def _log(self, msg):
        self.log_signal.emit(msg)

    def run(self):
        try:
            from huds_app.core.config import load_config, AppConfig
            from huds_app.interface.workflow import init_run, show_status
            from huds_app.sampling.huds import run_huds_sampling
            from huds_app.data.validation import import_labels as il
            from huds_app.model.train import train_model

            config = self.config_dict
            project_name = config.get("project_name", time.strftime("%Y%m%d_%H%M%S"))
            runs_dir = os.path.join(self.project_dir, "HUDS_runs")
            run_dir = os.path.join(runs_dir, project_name)
            os.makedirs(run_dir, exist_ok=True)
            config_path = os.path.join(run_dir, "config.json")

            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            time.sleep(0.5)

            self._log(f"Configuration saved to {config_path}")
            init_run(config_path, run_dir)
            self._log("Run initialized successfully")
            show_status(run_dir)

            r2_history = []
            max_steps = config.get("training", {}).get("max_steps", 3)

            for step in range(1, max_steps + 1):
                if self._abort:
                    self._log("Workflow aborted by user")
                    return

                self._log(f"{'='*40}")
                self._log(f"Step {step}: HUDS Active Sampling")
                self._log(f"{'='*40}")

                cfg = load_config(config_path)
                result = run_huds_sampling(run_dir, cfg, step)
                n_selected = len(result["selected_ids"])
                self._log(f"  Selected {n_selected} samples via HUDS")

                request_path = os.path.join(
                    run_dir, "requests", f"train_step_{step:03d}_request.csv"
                )
                self._log(f"  Request CSV generated ({n_selected} rows)")

                force_csv = self._run_maxwell_sweep(
                    request_path, config_path, self.aedt_path, self.design_name, step, run_dir
                )

                label_path, out_names = self._extract_labels(
                    force_csv, step, run_dir, cfg
                )

                if hasattr(cfg.model, "output_names"):
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

                imported = il(run_dir, step, label_path, overwrite=True)
                self._log(f"  Imported {imported} labels")

                last_print_pct = [0]
                def progress_cb(msg, percent):
                    if percent - last_print_pct[0] >= 10 or percent == 100:
                        self._log(f"  [Train] {msg}")
                        last_print_pct[0] = percent

                metrics = train_model(run_dir, cfg, progress_cb=progress_cb)
                r2 = metrics.get("val_r2_avg", float("nan"))
                r2_history.append(r2)

                import math
                r2_str = f"{r2:.4f}" if not (isinstance(r2, float) and math.isnan(r2)) else "N/A"
                self._log(f"  Training complete: val_r2_avg = {r2_str}")

                step_progress = int((step / max_steps) * 100)
                self.progress_signal.emit(step_progress)
                self.step_done_signal.emit(step, metrics)

                show_status(run_dir)

            self._log("All steps completed successfully")
            self.finished_all_signal.emit(r2_history)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.error_signal.emit(f"{e}\n{tb}")

    def _run_maxwell_sweep(self, request_csv, cfg_path, aedt_path, dsgn_name, step, run_dir):
        self._log("  Running Maxwell simulation...")
        env = os.environ.copy()
        env["ANSYSLMD_LICENSE_FILE"] = "24500@licensing.hkust.edu.cn"
        env["PYTHONPATH"] = PROJECT_ROOT + ";" + env.get("PYTHONPATH", "")

        sweep_script = os.path.join(
            PROJECT_ROOT, "huds_app", "interface", "maxwell_sweep.py"
        )

        result = subprocess.run(
            [sys.executable, sweep_script,
             "--csv", request_csv,
             "--config", cfg_path,
             "--project", aedt_path,
             "--design", dsgn_name,
             "--output-dir", os.path.dirname(aedt_path) if os.path.isfile(aedt_path) else aedt_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=600,
            env=env,
        )

        if result.returncode != 0:
            err = result.stderr.replace("\ufffd", "?")[:2000]
            self._log(f"  Simulation failed:\n{err}")
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
        raise RuntimeError("Simulation result file not found")

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
            raise RuntimeError("Could not extract data from simulation results")

        force_df = pd.DataFrame(force_data)
        pivoted = force_df.pivot(index="v_rounded", columns="output_name", values="value").reset_index()

        merged = request_df.copy()
        merged["v_rounded"] = merged[var_name].round(4)
        merged = merged.merge(pivoted, on="v_rounded", how="left")
        merged = merged.drop(columns=["v_rounded"])

        label_path = os.path.join(run_dir, "data", f"step{step}_labels.csv")
        write_csv(merged[[SAMPLE_ID_COLUMN, var_name] + out_names], label_path)
        return label_path, out_names
