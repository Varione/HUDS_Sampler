from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from huds_app.metrics import compute_metrics
from huds_app.model import ResidualMLP, build_model
from huds_app.storage import RunState, append_csv, ensure_run_dir, read_csv


def compute_normalization(train_df: pd.DataFrame, var_cols: list[str], out_cols: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    variable_stats = {}
    output_stats = {}

    for column in var_cols:
        values = train_df[column].astype(float)
        min_value = float(values.min())
        max_value = float(values.max())
        variable_stats[column] = {
            "min": min_value,
            "max": max_value,
            "range": float(max_value - min_value),
        }

    for column in out_cols:
        values = train_df[column].astype(float)
        std_value = float(values.std(ddof=0))
        output_stats[column] = {
            "mean": float(values.mean()),
            "std": std_value,
        }

    return {"variables": variable_stats, "outputs": output_stats}


def save_normalization(stats: dict[str, Any], path: str | Path) -> None:
    stats_path = Path(path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2)
        file.write("\n")


def load_normalization(path: str | Path) -> dict[str, Any]:
    stats_path = Path(path)
    with stats_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def apply_normalization(x: Any, var_stats: dict[str, float]) -> Any:
    value_range = float(var_stats["range"])
    if value_range == 0.0:
        if isinstance(x, torch.Tensor):
            return torch.zeros_like(x)
        return np.zeros_like(np.asarray(x, dtype=float))
    return (x - float(var_stats["min"])) / value_range


def apply_target_normalization(y: Any, out_stats: dict[str, float]) -> Any:
    std_value = float(out_stats["std"])
    if std_value == 0.0:
        if isinstance(y, torch.Tensor):
            return torch.zeros_like(y)
        return np.zeros_like(np.asarray(y, dtype=float))
    return (y - float(out_stats["mean"])) / std_value


def train_step(
    model: ResidualMLP, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    config: Any, 
    device: torch.device | str, 
    normalization_stats: dict[str, Any] | None = None  # FIX 5: Pass normalization for physical unit metrics
) -> dict[str, float]:
    start_time = time.perf_counter()
    device = torch.device(device)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    loss_fn = torch.nn.MSELoss()
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_metrics: dict[str, float] = {}
    epoch_metrics: dict[str, float] = {}

    for epoch in range(1, config.training.epochs_per_step + 1):
        model.train()
        train_loss_sum = 0.0
        train_sample_count = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            optimizer.step()

            batch_size = batch_x.size(0)
            train_loss_sum += float(loss.item()) * batch_size
            train_sample_count += batch_size

        train_loss = train_loss_sum / train_sample_count if train_sample_count else 0.0
        
        # FIX 5: Pass normalization stats to evaluation for physical unit metrics
        val_loss, val_metrics = _evaluate_model(
            model, 
            val_loader, 
            loss_fn, 
            config.model.output_names, 
            device,
            normalization=normalization_stats
        )
        
        epoch_metrics = {
            "step": float(getattr(config, "current_step", 0)),
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_r2_avg": float(val_metrics.get("r2_avg", 0.0)),
            "elapsed_s": float(time.perf_counter() - start_time),
        }
        
        # Record normalized metrics (always present)
        for output_name in config.model.output_names:
            epoch_metrics[f"val_r2_{output_name}"] = float(val_metrics.get(f"r2_{output_name}", 0.0))
            epoch_metrics[f"val_rmse_{output_name}"] = float(val_metrics.get(f"rmse_{output_name}", 0.0))
            
        # FIX 5: Record physical unit metrics if available (RMSE/MAE in original scale)
        for output_name in config.model.output_names:
            physical_rmse_key = f"val_rmse_{output_name}_physical"
            physical_mae_key = f"val_mae_{output_name}_physical"
            
            if physical_rmse_key in val_metrics:
                epoch_metrics[physical_rmse_key] = float(val_metrics.get(physical_rmse_key))
            if physical_mae_key in val_metrics:
                epoch_metrics[physical_mae_key] = float(val_metrics.get(physical_mae_key))

        latest_path = getattr(config, "latest_checkpoint_path", None)
        if latest_path is not None:
            _save_checkpoint(model, optimizer, epoch_metrics, Path(latest_path))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_metrics = epoch_metrics.copy()
            best_path = getattr(config, "best_checkpoint_path", None)
            if best_path is not None:
                _save_checkpoint(model, optimizer, epoch_metrics, Path(best_path))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.training.patience:
                break

    return best_metrics or epoch_metrics


def train_model(run_dir: str | Path, config: Any) -> dict[str, float]:
    """Train the surrogate model.
    
    FIX 2: Load existing checkpoint before training (unless retrain_from_scratch is True).
    FIX 5: Pass normalization stats to evaluation for physical unit metrics reporting.
    """
    run_path = ensure_run_dir(str(run_dir))
    datasets_dir = run_path / "datasets"
    checkpoints_dir = run_path / "checkpoints"
    metrics_dir = run_path / "metrics"
    artifacts_dir = run_path / "artifacts"

    train_df = read_csv(datasets_dir / "train_labeled.csv")
    val_df = read_csv(datasets_dir / "validation_labeled.csv")
    var_cols = [variable.name for variable in config.variables]
    out_cols = list(config.model.output_names)
    _validate_columns(train_df, var_cols + out_cols, datasets_dir / "train_labeled.csv")
    _validate_columns(val_df, var_cols + out_cols, datasets_dir / "validation_labeled.csv")

    normalization_path = artifacts_dir / "normalization.json"
    if normalization_path.exists():
        normalization = load_normalization(normalization_path)
    else:
        normalization = compute_normalization(train_df, var_cols, out_cols)
        save_normalization(normalization, normalization_path)

    train_x, train_y = _make_arrays(train_df, var_cols, out_cols, normalization)
    val_x, val_y = _make_arrays(val_df, var_cols, out_cols, normalization)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=config.training.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)),
        batch_size=config.training.batch_size,
        shuffle=False,
    )

    checkpoint_latest = checkpoints_dir / "model_latest.pt"
    checkpoint_best = checkpoints_dir / "model_best.pt"
    config.current_step = _load_current_step(run_path)
    config.latest_checkpoint_path = str(checkpoint_latest)
    config.best_checkpoint_path = str(checkpoint_best)

    device = torch.device(config.training.device if torch.cuda.is_available() or config.training.device != "cuda" else "cpu")
    
    # FIX 2: Load existing checkpoint before training (unless retrain_from_scratch is True)
    model = build_model(config)
    if not getattr(config.training, 'retrain_from_scratch', False) and checkpoint_latest.exists():
        print(f"Loading existing checkpoint from {checkpoint_latest}")
        try:
            checkpoint_data = torch.load(checkpoint_latest, map_location=device)
            model.load_state_dict(checkpoint_data["model_state_dict"])
            print("Checkpoint loaded successfully")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint ({e}), training from scratch")
    
    # FIX 5: Pass normalization stats so evaluation can compute physical unit metrics
    metrics = train_step(model, train_loader, val_loader, config, device, normalization_stats=normalization)

    metrics_path = metrics_dir / "training_metrics.csv"
    append_csv(pd.DataFrame([metrics]), metrics_path)
    _update_state(run_path, checkpoint_latest, checkpoint_best)
    return metrics


def _evaluate_model(
    model: torch.nn.Module, 
    val_loader: DataLoader, 
    loss_fn: torch.nn.Module, 
    output_names: list[str], 
    device: torch.device,
    normalization: dict[str, Any] | None = None  # FIX 5: Optional normalization stats for denormalization
) -> tuple[float, dict[str, float]]:
    """Evaluate model on validation set.
    
    FIX 5: When normalization stats are provided, compute metrics in both normalized 
    and physical unit spaces (RMSE/MAE). Normalized loss is always returned for 
    optimization tracking since R² is scale-invariant.
    """
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    y_true_batches = []
    y_pred_batches = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)
            batch_size = batch_x.size(0)
            loss_sum += float(loss.item()) * batch_size
            sample_count += batch_size
            y_true_batches.append(batch_y.cpu().numpy())
            y_pred_batches.append(predictions.cpu().numpy())

    val_loss = loss_sum / sample_count if sample_count else 0.0
    
    if not y_true_batches:
        return val_loss, compute_metrics(np.empty((0, len(output_names))), np.empty((0, len(output_names))), output_names)
    
    y_true_norm = np.vstack(y_true_batches)
    y_pred_norm = np.vstack(y_pred_batches)
    
    # Always compute normalized metrics first (for loss tracking and R² which is scale-invariant)
    metrics = compute_metrics(y_true_norm, y_pred_norm, output_names)
    
    # FIX 5: If normalization stats available, denormalize predictions and compute physical unit metrics
    if normalization and "outputs" in normalization:
        output_stats = normalization["outputs"]
        
        # Denormalize predictions to physical units for RMSE/MAE reporting
        y_true_physical = np.zeros_like(y_pred_norm)
        y_pred_physical = np.zeros_like(y_pred_norm)
        
        for i, name in enumerate(output_names):
            if name in output_stats:
                mean_val = float(output_stats[name]["mean"])
                std_val = float(output_stats[name]["std"])
                # Only denormalize if we have valid stats (avoid division by zero)
                if std_val > 0:
                    y_pred_physical[:, i] = y_pred_norm[:, i] * std_val + mean_val
                    y_true_physical[:, i] = y_true_norm[:, i] * std_val + mean_val
        
        # Compute physical unit metrics and add to results with _physical suffix
        for i, name in enumerate(output_names):
            if name in output_stats:
                mean_val = float(output_stats[name]["mean"])
                std_val = float(output_stats[name]["std"])
                if std_val > 0:
                    # Physical RMSE (meaningful absolute error)
                    physical_rmse = np.sqrt(np.mean((y_true_physical[:, i] - y_pred_physical[:, i]) ** 2))
                    metrics[f"val_rmse_{name}_physical"] = float(physical_rmse)
                    
                    # Physical MAE  
                    physical_mae = np.mean(np.abs(y_true_physical[:, i] - y_pred_physical[:, i]))
                    metrics[f"val_mae_{name}_physical"] = float(physical_mae)
    
    return val_loss, metrics


def _make_arrays(df: pd.DataFrame, var_cols: list[str], out_cols: list[str], normalization: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    x_columns = []
    for column in var_cols:
        values = df[column].to_numpy(dtype=np.float32)
        x_columns.append(apply_normalization(values, normalization["variables"][column]))

    y_columns = []
    for column in out_cols:
        values = df[column].to_numpy(dtype=np.float32)
        y_columns.append(apply_target_normalization(values, normalization["outputs"][column]))

    return np.column_stack(x_columns).astype(np.float32), np.column_stack(y_columns).astype(np.float32)


def _save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, metrics: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def _validate_columns(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")


def _load_current_step(run_path: Path) -> int:
    try:
        return RunState.load(str(run_path)).current_step
    except FileNotFoundError:
        return 0


def _update_state(run_path: Path, latest_checkpoint: Path, best_checkpoint: Path) -> None:
    try:
        state = RunState.load(str(run_path))
    except FileNotFoundError:
        state = RunState(run_dir=str(run_path))
    state.latest_checkpoint = str(latest_checkpoint)
    state.best_checkpoint = str(best_checkpoint)
    state.save()
