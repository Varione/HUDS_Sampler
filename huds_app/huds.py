from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

from huds_app.config import AppConfig, load_config
from huds_app.data_schema import SAMPLE_ID_COLUMN, STATUS_COLUMN
from huds_app.model import build_model
from huds_app.storage import RunState, ensure_run_dir, read_csv, write_csv


def mc_dropout_predict(model, x, repeat_times, batch_size):
    """Run MC Dropout prediction and return repeat predictions plus variance uncertainty."""
    if repeat_times <= 0:
        raise ValueError("repeat_times must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    was_training = model.training
    model.train()

    if isinstance(x, torch.Tensor):
        x_tensor = x.detach()
    else:
        x_tensor = torch.as_tensor(x, dtype=torch.float32)

    model_device = next(model.parameters()).device
    n_samples = x_tensor.shape[0]
    repeat_predictions = []

    with torch.no_grad():
        for _ in range(repeat_times):
            batch_predictions = []
            for start in range(0, n_samples, batch_size):
                batch = x_tensor[start : start + batch_size].to(model_device)
                prediction = model(batch)
                batch_predictions.append(prediction.detach().cpu())
            repeat_predictions.append(torch.cat(batch_predictions, dim=0))

    predictions = torch.stack(repeat_predictions, dim=0).numpy()
    uncertainties = predictions.var(axis=0).mean(axis=1)

    if not was_training:
        model.eval()

    return predictions, uncertainties


def select_huds(model, train_pool_df, unlabeled_mask, train_labeled_df, config, var_cols, device):
    """Select the next active learning batch with Hybrid Uncertainty and Diversity Sampling."""
    _validate_selection_inputs(train_pool_df, var_cols)

    unlabeled_pool = _filter_unlabeled_pool(train_pool_df, unlabeled_mask, train_labeled_df)
    if unlabeled_pool.empty:
        return _empty_selection_result()

    candidate_pool = _apply_pre_sampling(unlabeled_pool, config)
    n_select = min(int(config.training.sample_per_step), len(candidate_pool))
    if n_select <= 0:
        return _empty_selection_result()

    feature_values = candidate_pool[var_cols].to_numpy(dtype=np.float32)
    x_tensor = torch.tensor(feature_values, dtype=torch.float32, device=device)
    _, uncertainties = mc_dropout_predict(
        model,
        x_tensor,
        repeat_times=int(config.huds.repeat_times),
        batch_size=int(config.huds.batch_size),
    )

    topk_ratio = float(config.huds.topk_ratio)
    n_topk = min(len(candidate_pool), max(1, int(topk_ratio * len(candidate_pool))))
    topk_positions = np.argsort(-uncertainties)[:n_topk]
    topk_df = candidate_pool.iloc[topk_positions].copy()
    topk_uncertainties = uncertainties[topk_positions]
    n_clusters = min(n_select, n_topk)

    standardized_topk = _standardize(topk_df[var_cols].to_numpy(dtype=np.float32))
    labels = _cluster_topk(standardized_topk, n_clusters, config)
    selected_positions, cluster_stats = _select_cluster_representatives(
        topk_df,
        topk_uncertainties,
        labels,
        n_clusters,
    )

    # FIX 6: Ensure sample IDs are integers for type consistency
    selected_ids = [int(_sample_id(candidate_pool, position)) for position in selected_positions]
    selected_uncertainties = [float(uncertainties[position]) for position in selected_positions]

    if len(selected_ids) < n_select:
        standardized_pool = _standardize(candidate_pool[var_cols].to_numpy(dtype=np.float32))
        fallback_positions = _k_center_fill(
            standardized_pool,
            selected_positions,
            n_select,
        )
        for position in fallback_positions:
            selected_positions.append(position)
            # FIX 6: Ensure sample IDs are integers for type consistency
            selected_ids.append(int(_sample_id(candidate_pool, position)))
            selected_uncertainties.append(float(uncertainties[position]))

    return {
        "selected_ids": selected_ids,
        "uncertainties": selected_uncertainties,
        "topk_size": int(n_topk),
        "n_clusters": int(n_clusters),
        "cluster_stats": cluster_stats,
        "checkpoint_used": "",
    }


def run_huds_sampling(run_dir, config, step):
    """Run the full HUDS sampling workflow for one active learning step."""
    run_path = ensure_run_dir(str(run_dir))
    state = RunState.load(str(run_path))
    app_config = _resolve_config(run_path, config)
    _validate_sampling_step(state, app_config, step)
    variable_columns = [variable.name for variable in app_config.variables]

    train_pool_df = read_csv(run_path / "train_pool.csv")
    train_labeled_path = run_path / "datasets" / "train_labeled.csv"
    train_labeled_df = read_csv(train_labeled_path) if train_labeled_path.exists() else pd.DataFrame({SAMPLE_ID_COLUMN: []})

    checkpoint_path = _latest_checkpoint_path(run_path, state)
    normalized_pool_df = _normalize_pool_for_model(train_pool_df, run_path, variable_columns)
    unlabeled_mask = _build_unlabeled_mask(train_pool_df, train_labeled_df, state)

    device_obj = torch.device(app_config.training.device if torch.cuda.is_available() or app_config.training.device == "cpu" else "cpu")
    model = build_model(app_config).to(device_obj)
    _load_checkpoint_weights(model, checkpoint_path, device_obj)

    result = select_huds(
        model=model,
        train_pool_df=normalized_pool_df,
        unlabeled_mask=unlabeled_mask,
        train_labeled_df=train_labeled_df,
        config=app_config,
        var_cols=variable_columns,
        device=device_obj,
    )
    result["checkpoint_used"] = str(checkpoint_path)

    selected_ids = result["selected_ids"]
    request_df = train_pool_df[train_pool_df[SAMPLE_ID_COLUMN].isin(selected_ids)].copy()
    request_df = _order_request_rows(request_df, selected_ids)
    request_path = run_path / "requests" / f"train_step_{int(step):03d}_request.csv"
    diagnostics_path = run_path / "artifacts" / f"huds_step_{int(step):03d}.json"

    write_csv(request_df, request_path)
    with diagnostics_path.open("w", encoding="utf-8") as file:
        json.dump(_json_ready(result), file, indent=2)
        file.write("\n")

    state.current_step = int(step)
    state.latest_checkpoint = str(checkpoint_path.relative_to(run_path)) if checkpoint_path.is_relative_to(run_path) else str(checkpoint_path)
    
    # FIX 1: Update pending_sample_ids and train_requests with proper status tracking
    state.pending_sample_ids.extend(selected_ids)
    state.train_requests[str(step)] = {
        "path": str(request_path.relative_to(run_path)),
        "status": "exported",  # Track request status for workflow progression
        "diagnostics": str(diagnostics_path.relative_to(run_path)),
    }
    state.save()

    return result


def _validate_sampling_step(state, config, step):
    requested_step = int(step)

    if requested_step <= 0:
        raise ValueError("sampling step must be >= 1")
    if requested_step > int(config.training.max_steps):
        raise ValueError(
            f"sampling step {requested_step} exceeds configured max_steps={int(config.training.max_steps)}"
        )
    if str(requested_step) in state.train_requests:
        raise ValueError(f"training request for step {requested_step} already exists")

    expected_step = int(state.current_step) + 1
    if requested_step != expected_step:
        raise ValueError(
            f"sampling step must be the next step in sequence: expected {expected_step}, got {requested_step}"
        )

    has_pending_train = any(
        request.get("status") in {"exported", "partial"}
        for request in state.train_requests.values()
    )
    if has_pending_train:
        raise ValueError("cannot sample a new step while previous training requests are still pending labels")

    if int(state.trained_step) < int(state.current_step):
        raise ValueError("cannot sample a new step before the current step has been trained")


def _validate_selection_inputs(train_pool_df, var_cols):
    missing_columns = [column for column in [SAMPLE_ID_COLUMN, *var_cols] if column not in train_pool_df.columns]
    if missing_columns:
        raise ValueError(f"train_pool_df missing required column(s): {missing_columns}")


def _filter_unlabeled_pool(train_pool_df, unlabeled_mask, train_labeled_df):
    mask = pd.Series(unlabeled_mask, index=train_pool_df.index).astype(bool)
    if STATUS_COLUMN in train_pool_df.columns:
        mask &= train_pool_df[STATUS_COLUMN].astype(str).eq("unlabeled")
    if SAMPLE_ID_COLUMN in train_labeled_df.columns and not train_labeled_df.empty:
        labeled_ids = set(train_labeled_df[SAMPLE_ID_COLUMN].dropna().tolist())
        mask &= ~train_pool_df[SAMPLE_ID_COLUMN].isin(labeled_ids)
    return train_pool_df.loc[mask].copy().reset_index(drop=True)


def _apply_pre_sampling(unlabeled_pool, config):
    pre_n = int(config.huds.pre_n)
    if pre_n <= 0 or pre_n >= len(unlabeled_pool):
        return unlabeled_pool.reset_index(drop=True)
    return unlabeled_pool.sample(n=pre_n, random_state=int(config.random_seed)).reset_index(drop=True)


def _standardize(values):
    means = values.mean(axis=0, keepdims=True)
    stds = values.std(axis=0, keepdims=True)
    stds[stds == 0.0] = 1.0
    return (values - means) / stds


def _cluster_topk(features, n_clusters, config):
    if n_clusters <= 1:
        return np.zeros(features.shape[0], dtype=np.int64)

    # Try FAISS first if configured
    if bool(config.huds.use_faiss):
        try:
            import faiss

            kmeans = faiss.Kmeans(
                d=features.shape[1],
                k=n_clusters,
                niter=25,
                seed=int(config.random_seed),
                verbose=False,
            )
            kmeans.train(features.astype(np.float32))
            _, labels = kmeans.index.search(features.astype(np.float32), 1)
            return labels.reshape(-1).astype(np.int64)
        except ImportError:
            pass

    # Fallback to sklearn KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=int(config.random_seed))
    return kmeans.fit_predict(features)


def _select_cluster_representatives(topk_df, topk_uncertainties, labels, n_clusters):
    selected_positions = []
    cluster_stats = {}
    for cluster_id in range(n_clusters):
        member_positions = np.flatnonzero(labels == cluster_id)
        if member_positions.size == 0:
            cluster_stats[str(cluster_id)] = {"size": 0, "selected_id": None, "max_uncertainty": None, "mean_uncertainty": None}
            continue

        local_best = member_positions[np.argmax(topk_uncertainties[member_positions])]
        
        # FIX 6: Ensure sample IDs are integers for type consistency  
        original_position = int(topk_df.index[local_best])
        selected_positions.append(original_position)
        cluster_uncertainties = topk_uncertainties[member_positions]
        cluster_stats[str(cluster_id)] = {
            "size": int(member_positions.size),
            # FIX 6: Ensure sample IDs are integers for type consistency
            "selected_id": int(_sample_id(topk_df, local_best)),
            "max_uncertainty": float(cluster_uncertainties.max()),
            "mean_uncertainty": float(cluster_uncertainties.mean()),
        }
    return selected_positions, cluster_stats


def _k_center_fill(features, selected_positions, n_select):
    selected_set = set(selected_positions)
    if not selected_set:
        selected_set.add(0)

    added_positions = []
    while len(selected_set) < n_select and len(selected_set) < features.shape[0]:
        selected_array = np.array(sorted(selected_set), dtype=int)
        distances = euclidean_distances(features, features[selected_array]).min(axis=1)
        for position in selected_set:
            distances[position] = -np.inf
        next_position = int(np.argmax(distances))
        selected_set.add(next_position)
        added_positions.append(next_position)
    return added_positions


def _sample_id(df, position):
    """Extract sample ID from DataFrame at given position.
    
    FIX 6: Force integer conversion for type consistency across state/diagnostics/CSV files.
    """
    value = df.iloc[int(position)][SAMPLE_ID_COLUMN]
    # Convert to int if it's a numeric value that represents an integer ID
    result = value.item() if hasattr(value, "item") else value
    try:
        return int(float(result))  # Handle both int and float representations
    except (ValueError, TypeError):
        return result


def _empty_selection_result():
    return {
        "selected_ids": [],
        "uncertainties": [],
        "topk_size": 0,
        "n_clusters": 0,
        "cluster_stats": {},
        "checkpoint_used": "",
    }


def _resolve_config(run_path: Path, config: AppConfig | str | Path | None) -> AppConfig:
    if isinstance(config, AppConfig):
        return config
    if config is None:
        return load_config(str(run_path / "config.json"))
    if isinstance(config, (str, Path)):
        return load_config(str(config))
    return cast(AppConfig, config)


def _latest_checkpoint_path(run_path: Path, state: RunState):
    checkpoint = state.latest_checkpoint or "checkpoints/model_latest.pt"
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = run_path / checkpoint_path
    if not checkpoint_path.exists():
        checkpoint_path = run_path / "checkpoints" / "model_latest.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _normalize_pool_for_model(train_pool_df, run_path, variable_columns):
    normalization_path = run_path / "artifacts" / "normalization.json"
    if not normalization_path.exists():
        raise FileNotFoundError(f"Normalization statistics not found: {normalization_path}")
    with normalization_path.open("r", encoding="utf-8") as file:
        normalization = json.load(file)

    var_stats = normalization.get("variables", {})
    normalized_df = train_pool_df.copy()
    for col in variable_columns:
        if col in var_stats and col in normalized_df.columns:
            values = normalized_df[col].to_numpy(dtype=np.float32)
            value_range = float(var_stats[col]["range"])
            min_val = float(var_stats[col]["min"])
            # Apply normalization safely
            if value_range > 0:
                normalized_df[col] = (values - min_val) / value_range
            else:
                normalized_df[col] = 0.0
    return normalized_df


def _build_unlabeled_mask(train_pool_df, train_labeled_df, state):
    mask = pd.Series(True, index=train_pool_df.index)
    if STATUS_COLUMN in train_pool_df.columns:
        mask &= train_pool_df[STATUS_COLUMN].astype(str).eq("unlabeled")
    if SAMPLE_ID_COLUMN in train_labeled_df.columns and not train_labeled_df.empty:
        mask &= ~train_pool_df[SAMPLE_ID_COLUMN].isin(train_labeled_df[SAMPLE_ID_COLUMN])
    blocked_ids = set(state.used_sample_ids) | set(state.pending_sample_ids)
    if blocked_ids:
        mask &= ~train_pool_df[SAMPLE_ID_COLUMN].isin(blocked_ids)
    return mask


def _load_checkpoint_weights(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)


def _order_request_rows(request_df, selected_ids):
    order = {sample_id: index for index, sample_id in enumerate(selected_ids)}
    ordered = request_df.copy()
    ordered["__huds_order"] = ordered[SAMPLE_ID_COLUMN].map(order)
    ordered = ordered.sort_values("__huds_order").drop(columns="__huds_order")
    return ordered.reset_index(drop=True)


def _json_ready(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
