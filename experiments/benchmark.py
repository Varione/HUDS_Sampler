"""HUDS vs Random Sampling Benchmark Orchestrator.

Runs all 9 scenarios x 2 strategies = 18 experiments, collecting per-step
metrics (R2, RMSE, labeled count, wall time) to CSV.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from huds_app.core.config import (
    AppConfig,
    CandidatePoolConfig,
    HUDSConfig,
    ModelConfig,
    TrainingConfig,
    ValidationConfig,
    VariableConfig,
)
from huds_app.core.storage import RunState, ensure_run_dir, write_csv
from huds_app.data.pool import create_candidate_pool, save_pool_files, split_pool
from huds_app.interface.workflow import evaluate as wf_evaluate
from huds_app.model.train import train_model
from huds_app.sampling.huds import run_huds_sampling

from experiments.ground_truth import generate_v2i, generate_v2ts, generate_v2v


# ---------------------------------------------------------------------------
# Scenario loading
# ---------------------------------------------------------------------------

def load_scenarios(path: str | Path) -> list[dict]:
    """Load scenarios from YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("scenarios", [])


def build_app_config(scenario: dict) -> AppConfig:
    """Convert scenario dict to AppConfig."""
    m = scenario.get("model", {})
    t = scenario.get("training", {})
    h = scenario.get("huds", {})
    cp = scenario.get("candidate_pool", {})

    return AppConfig(
        project_name=scenario.get("name", scenario.get("id", "")),
        random_seed=42,
        variables=[
            VariableConfig(
                name=v["name"],
                min=float(v["min"]),
                max=float(v["max"]),
                sample_points=int(v["sample_points"]),
            )
            for v in scenario.get("variables", [])
        ],
        candidate_pool=CandidatePoolConfig(
            total_samples=int(cp.get("total_samples", 1000)),
            train_ratio=float(cp.get("train_ratio", 0.8)),
            validation_ratio=float(cp.get("validation_ratio", 0.2)),
        ),
        model=ModelConfig(
            model_type=m.get("model_type", "vector_to_vector"),
            output_names=list(m.get("output_names", [])),
            hidden_dim=int(m.get("hidden_dim", 128)),
            encoder_blocks=int(m.get("encoder_blocks", 4)),
            dropout=float(m.get("dropout", 0.1)),
            seq_len=int(m.get("seq_len", 50)),
            decoder_layers=int(m.get("decoder_layers", 2)),
            img_h=int(m.get("img_h", 32)),
            img_w=int(m.get("img_w", 32)),
            channels=int(m.get("channels", 1)),
            decoder_blocks=int(m.get("decoder_blocks", 3)),
        ),
        validation=ValidationConfig(),
        training=TrainingConfig(
            initial_train_size=int(t.get("initial_train_size", 50)),
            sample_per_step=int(t.get("sample_per_step", 64)),
            max_steps=int(t.get("max_steps", 10)),
            epochs_per_step=int(t.get("epochs_per_step", 100)),
            batch_size=int(t.get("batch_size", 256)),
            learning_rate=float(t.get("learning_rate", 0.001)),
            patience=int(t.get("patience", 30)),
            device=str(t.get("device", "cuda")),
        ),
        huds=HUDSConfig(
            pre_n=int(h.get("pre_n", 0)),
            repeat_times=int(h.get("repeat_times", 30)),
            topk_ratio=float(h.get("topk_ratio", 0.6)),
            batch_size=int(h.get("batch_size", 256)),
            use_faiss=bool(h.get("use_faiss", False)),
            use_top_p=bool(h.get("use_top_p", False)),
            top_p_threshold=float(h.get("top_p_threshold", 0.9)),
        ),
    )


# ---------------------------------------------------------------------------
# Ground truth labeling
# ---------------------------------------------------------------------------

def generate_labels(scenario: dict, n: int) -> pd.DataFrame:
    """Generate ground truth labels for n samples matching scenario config."""
    m = scenario.get("model", {})
    model_type = m.get("model_type", "vector_to_vector")
    var_names = [v["name"] for v in scenario.get("variables", [])]
    input_dim = len(var_names)
    output_names = list(m.get("output_names", []))

    if model_type == "vector_to_time_series":
        seq_len = int(m.get("seq_len", 50))
        num_channels = len(output_names) // seq_len if output_names else 1
        df = generate_v2ts(n, input_dim, seq_len=seq_len, output_dim=num_channels)
    elif model_type == "vector_to_image":
        img_h = int(m.get("img_h", 32))
        img_w = int(m.get("img_w", 32))
        channels = int(m.get("channels", 1))
        df = generate_v2i(n, input_dim, img_h=img_h, img_w=img_w, channels=channels)
    else:
        # V2V
        output_dim = len(output_names)
        df = generate_v2v(n, input_dim, output_dim=output_dim)

    # Unified column renaming: extract output columns (exclude sample_id + var_names),
    # then rename to match scenario's output_names in order.
    exclude_cols = {"sample_id"} | set(var_names)
    output_cols = [col for col in df.columns if col not in exclude_cols]

    rename_map = {}
    for i, col in enumerate(output_cols):
        if i < len(output_names):
            rename_map[col] = output_names[i]
        else:
            rename_map[col] = f"extra_{i}"  # Should not happen if config is correct

    df = df.rename(columns=rename_map)
    return df


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    scenario: dict,
    strategy: str,
    results_dir: Path,
) -> list[dict]:
    """Run one scenario with one sampling strategy. Returns per-step metrics."""
    config = build_app_config(scenario)
    run_id = f"{scenario['id']}_{strategy}"
    run_path = results_dir / run_id

    # Clean previous run
    if run_path.exists():
        shutil.rmtree(run_path)

    var_names = [v.name for v in config.variables]
    output_names = list(config.model.output_names)
    total_pool = config.candidate_pool.total_samples

    # Step 1: Generate full labeled pool
    print(f"  Generating labels for {total_pool} samples...")
    full_labels = generate_labels(scenario, total_pool)

    # Step 2: Initialize run (creates pools)
    ensure_run_dir(str(run_path))
    pool_df = create_candidate_pool(config)
    train_df, valid_df = split_pool(pool_df, config, config.random_seed)
    save_pool_files(pool_df, train_df, valid_df, run_path)

    # Save config as JSON for workflow compatibility
    import json
    from dataclasses import asdict

    def to_serializable(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: to_serializable(v) for k, v in asdict(obj).items()}
        if isinstance(obj, list):
            return [to_serializable(i) for i in obj]
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        return obj

    config_json = to_serializable(config)
    with open(run_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_json, f, indent=2)

    # Step 3: Label validation set
    valid_sample_ids = set(valid_df["sample_id"].tolist())
    valid_labeled = full_labels[full_labels["sample_id"].isin(valid_sample_ids)].copy()
    # Keep only required columns
    valid_labeled = valid_labeled[["sample_id"] + var_names + output_names]
    write_csv(valid_labeled, run_path / "datasets" / "validation_labeled.csv")

    # Initialize state
    state = RunState(run_dir=str(run_path))
    state.validation_request_created = True
    state.validation_labeled = True
    state.save()

    results = []
    max_steps = config.training.max_steps

    for step in range(max_steps + 1):
        t_start = time.perf_counter()

        # Select samples for this step
        if step == 0:
            # Initial training set - random subset of train pool
            n_init = config.training.initial_train_size
            train_ids = train_df["sample_id"].tolist()[:n_init]
        else:
            if strategy == "huds":
                # Run HUDS sampling
                try:
                    huds_result = run_huds_sampling(run_path, config, step)
                    train_ids = huds_result.get("selected_ids", [])
                except Exception as e:
                    print(f"  HUDS sampling failed at step {step}: {e}")
                    break
            else:
                # Random sampling
                state = RunState.load(str(run_path))
                labeled_ids = set(state.used_sample_ids)
                available = [sid for sid in train_df["sample_id"].tolist() if sid not in labeled_ids]
                n_select = min(config.training.sample_per_step, len(available))
                rng = np.random.default_rng(config.random_seed + step)
                train_ids = rng.choice(available, size=n_select, replace=False).tolist()

        # Label selected samples
        step_labels = full_labels[full_labels["sample_id"].isin(set(train_ids))].copy()
        step_labels = step_labels[["sample_id"] + var_names + output_names]

        # Update train_labeled.csv (cumulative)
        existing_path = run_path / "datasets" / "train_labeled.csv"
        if existing_path.exists():
            existing = pd.read_csv(existing_path)
            combined = pd.concat([existing, step_labels], ignore_index=True)
            combined = combined.drop_duplicates(subset="sample_id", keep="last")
        else:
            combined = step_labels
        write_csv(combined, run_path / "datasets" / "train_labeled.csv")

        # Train model
        config.latest_checkpoint_path = str(run_path / "checkpoints" / "model_latest.pt")
        config.best_checkpoint_path = str(run_path / "checkpoints" / "model_best.pt")
        config.current_step = step

        try:
            train_metrics = train_model(run_path, config)
        except Exception as e:
            print(f"  Training failed at step {step}: {e}")
            break

        # Evaluate
        t_train = time.perf_counter() - t_start

        # Update state
        state = RunState.load(str(run_path))
        state.current_step = step
        state.trained_step = step
        state.used_sample_ids.extend(train_ids)
        state.train_requests[str(step)] = {
            "path": f"datasets/train_labeled.csv",
            "status": "labeled",
            "diagnostics": "",
        }
        state.save()

        # Record metrics
        labeled_count = len(combined)
        step_result = {
            "scenario_id": scenario["id"],
            "model_type": config.model.model_type,
            "strategy": strategy,
            "step": step,
            "labeled_count": labeled_count,
            "val_r2_avg": train_metrics.get("val_r2_avg", 0.0),
            "val_loss": train_metrics.get("val_loss", float("inf")),
            "elapsed_s": round(t_train, 2),
        }

        # Per-output metrics
        for out_name in output_names[:3]:  # First 3 outputs only
            r2_key = f"val_r2_{out_name}"
            rmse_key = f"val_rmse_{out_name}"
            step_result[r2_key] = train_metrics.get(r2_key, 0.0)
            step_result[rmse_key] = train_metrics.get(rmse_key, float("inf"))

        results.append(step_result)
        print(f"  Step {step}: labeled={labeled_count}, R2={step_result['val_r2_avg']:.4f}, time={t_train:.1f}s")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HUDS vs Random Sampling Benchmark")
    parser.add_argument("--scenarios", default=str(PROJECT_ROOT / "experiments" / "scenarios.yaml"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "experiments" / "results" / "benchmark_results.csv"))
    parser.add_argument("--strategy", default="both", choices=["huds", "random", "both"])
    parser.add_argument("--scenario-filter", default=None, help="Run only specific scenario ID")
    args = parser.parse_args()

    scenarios = load_scenarios(args.scenarios)
    if args.scenario_filter:
        scenarios = [s for s in scenarios if s["id"] == args.scenario_filter]

    results_dir = Path(args.output).parent
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.strategy == "both":
        strategies = ["random", "huds"]
    else:
        strategies = [args.strategy]

    all_results = []
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario['id']} ({scenario.get('name', '')})")
        print(f"{'='*60}")
        for strategy in strategies:
            print(f"\n  Strategy: {strategy.upper()}")
            try:
                step_results = run_experiment(scenario, strategy, results_dir)
                all_results.extend(step_results)
            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()

    # Save results
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(args.output, index=False)
    print(f"\n{'='*60}")
    print(f"Benchmark complete. Results saved to: {args.output}")
    print(f"Total experiments: {len(set(zip(df_results['scenario_id'], df_results['strategy'])))}")
    print(f"Total data points: {len(df_results)}")


if __name__ == "__main__":
    main()
