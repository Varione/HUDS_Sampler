from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from huds_app.core.metrics import compute_metrics
from huds_app.model.architecture import build_model
from huds_app.core.storage import RunState, append_csv, ensure_run_dir, read_csv, resolve_device


def compute_normalization(train_df: pd.DataFrame, var_cols: list[str], out_cols: list[str], variable_configs: dict[str, dict[str, float]] | None = None) -> dict[str, dict[str, dict[str, float]]]:
    """Compute normalization statistics.

    FIX 11: If variable_configs is provided, use configured physical ranges
    instead of data-derived ranges to handle OOD candidates properly.
    """
    variable_stats = {}
    output_stats = {}

    for column in var_cols:
        values = train_df[column].astype(float)
        if variable_configs and column in variable_configs:
            min_value = float(variable_configs[column]["min"])
            max_value = float(variable_configs[column]["max"])
        else:
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
    # FIX 18: Use tolerance comparison to catch near-zero range (IEEE 754 precision)
    if abs(value_range) < 1e-12:
        if isinstance(x, torch.Tensor):
            return torch.zeros_like(x)
        return np.zeros_like(np.asarray(x, dtype=float))
    return (x - float(var_stats["min"])) / value_range


def apply_target_normalization(y: Any, out_stats: dict[str, float]) -> Any:
    std_value = float(out_stats["std"])
    # FIX 18: Use tolerance comparison to catch near-zero std (IEEE 754 precision)
    if abs(std_value) < 1e-12:
        if isinstance(y, torch.Tensor):
            return torch.zeros_like(y)
        return np.zeros_like(np.asarray(y, dtype=float))
    return (y - float(out_stats["mean"])) / std_value


def _reshape_target(y: torch.Tensor, config: Any) -> torch.Tensor:
    """Reshape flat target tensor to match model's multi-dimensional output.

    V2V: No reshape needed (batch, output_dim)
    V2TS: Reshape from (batch, seq_len * channels) to (batch, seq_len, channels)
    V2I:  Reshape from (batch, H*W*C) to (batch, C, H, W)
    """
    model_type = config.model.model_type

    if model_type == "vector_to_time_series":
        seq_len = config.model.seq_len
        channels = len(config.model.output_names) // seq_len
        return y.reshape(-1, seq_len, channels)

    if model_type == "vector_to_image":
        return y.reshape(-1, config.model.channels, config.model.img_h, config.model.img_w)

    # V2V: no reshape needed
    return y


def train_step(
    model: ResidualMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Any,
    device: torch.device | str,
    normalization_stats: dict[str, Any] | None = None,  # FIX 5: Pass normalization for physical unit metrics
    optimizer: torch.optim.Optimizer | None = None,       # P2-Fix D: Optional pre-built optimizer (checkpoint restore)
    progress_cb: Any | None = None,
) -> dict[str, float]:
    """Execute one or more training steps.

    P2-Fix D: If an optimizer instance is passed (from checkpoint restore), reuse it
              for continuous training momentum. Otherwise create a fresh Adam optimizer.
    """
    start_time = time.perf_counter()
    device = torch.device(device)
    model.to(device)

    # Mixed precision: enable only on CUDA
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    loss_fn = torch.nn.MSELoss()
    best_val_loss = float("inf")
    best_val_r2 = -float("inf")
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

            # Reshape target to match model output shape for multi-dimensional models
            batch_y = _reshape_target(batch_y, config)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                predictions = model(batch_x)
                loss = loss_fn(predictions, batch_y)

            if scaler is not None:
                scaler.scale(loss).backward()
                # FIX 12: Gradient clipping for training stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                # FIX 12: Gradient clipping for training stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            normalization=normalization_stats,
            config=config,
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

        current_r2 = float(val_metrics.get("r2_avg", 0.0))
        r2_valid = not np.isnan(current_r2)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if r2_valid:
                best_val_r2 = current_r2
            epochs_without_improvement = 0
            best_metrics = epoch_metrics.copy()
            best_path = getattr(config, "best_checkpoint_path", None)
            if best_path is not None:
                _save_checkpoint(model, optimizer, epoch_metrics, Path(best_path))
        else:
            if r2_valid:
                r2_regressed = current_r2 < best_val_r2 - 0.005
                if r2_regressed:
                    epochs_without_improvement += 1
                elif current_r2 >= best_val_r2:
                    best_val_r2 = current_r2
                    best_path = getattr(config, "best_checkpoint_path", None)
                    if best_path is not None:
                        _save_checkpoint(model, optimizer, epoch_metrics, Path(best_path))
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= config.training.patience:
                r2_str = f"{current_r2:.4f}" if r2_valid else "N/A"
                if progress_cb is not None:
                    progress_cb(
                        f"Early stopping at epoch {epoch}/{config.training.epochs_per_step} "
                        f"(val_loss={val_loss:.6f}, val_r2={r2_str})",
                        100,
                    )
                break

        if progress_cb is not None:
            percent = int(epoch / max(1, config.training.epochs_per_step) * 100)
            r2_str = f"{current_r2:.4f}" if r2_valid else "N/A"
            progress_cb(
                (
                    f"Epoch {epoch}/{config.training.epochs_per_step} | "
                    f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                    f"R2={r2_str}"
                ),
                min(100, max(0, percent)),
            )

    return best_metrics or epoch_metrics


def train_model(run_dir: str | Path, config: Any, progress_cb: Any | None = None) -> dict[str, float]:
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

    val_path = datasets_dir / "val_labeled.csv"
    if val_path.exists():
        val_df = read_csv(val_path)
    else:
        # Fallback: use a random subset of train data for validation
        rng = np.random.RandomState(config.random_seed)
        n_val = max(1, int(len(train_df) * 0.2))
        val_idx = rng.choice(len(train_df), size=n_val, replace=False)
        val_df = train_df.iloc[val_idx].copy()
        print(f"  Warning: val_labeled.csv not found, using {n_val} train samples as validation")
    var_cols = [variable.name for variable in config.variables]
    out_cols = list(config.model.output_names)
    _validate_columns(train_df, var_cols + out_cols, datasets_dir / "train_labeled.csv")
    _validate_columns(val_df, var_cols + out_cols, datasets_dir / "val_labeled.csv")

    normalization_path = artifacts_dir / "normalization.json"
    if normalization_path.exists():
        normalization = load_normalization(normalization_path)
    else:
        # FIX 11: Use configured physical ranges for variable normalization
        var_configs = {v.name: {"min": v.min, "max": v.max} for v in config.variables}
        normalization = compute_normalization(train_df, var_cols, out_cols, var_configs)
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

    device = resolve_device(config.training.device)

    # FIX 2: Load existing checkpoint before training (unless retrain_from_scratch is True)
    model = build_model(config).to(device)
    restored_optimizer: torch.optim.Optimizer | None = None

    if not getattr(config.training, 'retrain_from_scratch', False) and checkpoint_latest.exists():
        print(f"Loading existing checkpoint from {checkpoint_latest}")
        try:
            checkpoint_data = torch.load(checkpoint_latest, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint_data["model_state_dict"])

            # P2-Fix D: Optionally restore optimizer state for continuous training momentum
            if getattr(config.training, 'restore_optimizer_state', True):
                optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
                optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
                _move_optimizer_to_device(optimizer, device)
                restored_optimizer = optimizer
                print("Checkpoint and optimizer state loaded successfully")
            else:
                print("Checkpoint loaded (optimizer reset per config)")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint ({e}), training from scratch")

    # FIX 5: Pass normalization stats so evaluation can compute physical unit metrics
    metrics = train_step(model, train_loader, val_loader, config, device,
                         normalization_stats=normalization, optimizer=restored_optimizer,
                         progress_cb=progress_cb)

    metrics_path = metrics_dir / "training_metrics.csv"
    append_csv(pd.DataFrame([metrics]), metrics_path)
    _update_state(run_path, checkpoint_latest, checkpoint_best, config)
    return metrics


def _evaluate_model(
    model: torch.nn.Module,
    val_loader: DataLoader,
    loss_fn: torch.nn.Module,
    output_names: list[str],
    device: torch.device,
    normalization: dict[str, Any] | None = None,
    config: Any | None = None,
) -> tuple[float, dict[str, float]]:
    """Evaluate model on validation set.

    Handles multi-dimensional outputs (V2TS: batch,seq,output / V2I: batch,ch,h,w)
    by flattening all spatial dimensions for per-output-name metrics.
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

            # Reshape target to match model output shape
            if config is not None:
                batch_y = _reshape_target(batch_y, config)

            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)
            batch_size = batch_x.size(0)
            loss_sum += float(loss.item()) * batch_size
            sample_count += batch_size
            y_true_batches.append(batch_y.cpu().numpy())
            y_pred_batches.append(predictions.cpu().numpy())

    val_loss = loss_sum / sample_count if sample_count else 0.0

    if not y_true_batches:
        return val_loss, compute_metrics(
            np.empty((0, len(output_names))),
            np.empty((0, len(output_names))),
            output_names,
        )

    y_true_all = np.concatenate(y_true_batches, axis=0)
    y_pred_all = np.concatenate(y_pred_batches, axis=0)

    # Flatten to (batch, -1) for metrics: each output_name occupies contiguous columns
    batch_size = y_true_all.shape[0]
    y_true_flat = y_true_all.reshape(batch_size, -1)
    y_pred_flat = y_pred_all.reshape(batch_size, -1)

    # Split flattened array by output_names (equal chunks)
    metrics = compute_metrics(y_true_flat, y_pred_flat, output_names)

    # Physical unit metrics
    if normalization and "outputs" in normalization:
        output_stats = normalization["outputs"]
        for i, name in enumerate(output_names):
            if name not in output_stats:
                continue
            mean_val = float(output_stats[name]["mean"])
            std_val = float(output_stats[name]["std"])
            if std_val <= 0:
                continue

            cols = _output_slice(y_true_flat, i, len(output_names))
            y_true_phys = y_true_flat[:, cols] * std_val + mean_val
            y_pred_phys = y_pred_flat[:, cols] * std_val + mean_val

            physical_rmse = np.sqrt(np.mean((y_true_phys - y_pred_phys) ** 2))
            metrics[f"val_rmse_{name}_physical"] = float(physical_rmse)
            physical_mae = np.mean(np.abs(y_true_phys - y_pred_phys))
            metrics[f"val_mae_{name}_physical"] = float(physical_mae)

    return val_loss, metrics


def _output_slice(flat: np.ndarray, idx: int, total_outputs: int) -> slice:
    """Return column slice for output index idx in a flattened (batch, total_elements) array."""
    n_cols = flat.shape[1]
    if n_cols % total_outputs != 0:
        raise ValueError(
            f"flat columns={n_cols} not divisible by total_outputs={total_outputs}. "
            f"Check that len(output_names) is divisible by seq_len for vector_to_time_series models."
        )
    chunk = n_cols // total_outputs
    return slice(idx * chunk, (idx + 1) * chunk)


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
    """Save checkpoint with full metadata for reproducibility and recovery."""
    import sys
    import os
    import platform

    path.parent.mkdir(parents=True, exist_ok=True)

    # FIX 18: Include comprehensive metadata for reproducibility
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "format_version": 1,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "os_platform": platform.platform(),
        "git_commit": None,  # Could be added if repository is available
    }

    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__)),  # Assume repo root relative to train.py
        )
        if result.returncode == 0:
            checkpoint["git_commit"] = result.stdout.strip()
    except Exception:
        pass

    torch.save(checkpoint, path)


def _move_optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _validate_columns(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")


def _load_current_step(run_path: Path) -> int:
    try:
        return RunState.load(str(run_path)).current_step
    except FileNotFoundError:
        return 0


def _update_state(run_path: Path, latest_checkpoint: Path, best_checkpoint: Path, config: Any) -> None:
    try:
        state = RunState.load(str(run_path))
    except FileNotFoundError:
        state = RunState(run_dir=str(run_path))
    state.latest_checkpoint = str(latest_checkpoint)
    state.best_checkpoint = str(best_checkpoint)
    state.trained_step = int(getattr(config, "current_step", state.current_step))
    state.save()