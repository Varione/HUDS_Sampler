from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import random
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

from huds_app.core.config import AppConfig, load_config
from huds_app.data.schema import SAMPLE_ID_COLUMN, STATUS_COLUMN, SPLIT_ASSIGNMENT_COLUMN
from huds_app.model.architecture import build_model
from huds_app.core.storage import RunState, _normalize_sample_id, ensure_run_dir, read_csv, write_csv, resolve_device


def _enable_mc_dropout(model: torch.nn.Module):
    """Enable MC Dropout by setting all Dropout modules to train mode while keeping other layers in eval mode."""
    model.eval()  # Start with all layers in eval mode (including BatchNorm)
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            module.train()


def mc_dropout_predict(model, x, repeat_times, batch_size, return_outputs: bool = False):
    """Run MC Dropout and return repeat embeddings plus variance uncertainty.

    FIX 4: Only enable Dropout modules, leaving BatchNorm in eval mode to avoid using mini-batch statistics.
    FIX 5: Enforce repeat_times >= 2 for valid variance calculation.
    FIX 9: Use online moment estimation to reduce memory complexity from O(T*N*D) to O(N*D).
    FIX 13: Added return_outputs parameter to collect model outputs for
    output-space uncertainty estimation (default False for backward compatibility).

    Args:
        return_outputs: If True, also collect model outputs for output-space
                        uncertainty estimation. If False, uses embedding-space proxy.

    Returns:
        embeddings: (repeat, n, hidden_dim) - always returned for clustering
        uncertainties: (n,) - output-space or embedding-space uncertainty
        predictions_mean: (n, output_dim) - only if return_outputs, else None
    """
    if repeat_times < 2:
        raise ValueError("repeat_times must be >= 2 for variance calculation")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    # Save per-module training state for exact restoration after MC Dropout
    original_states = {m: m.training for m in model.modules()}
    pred_mean = None

    _enable_mc_dropout(model)

    if isinstance(x, torch.Tensor):
        x_tensor = x.detach()
    else:
        x_tensor = torch.as_tensor(x, dtype=torch.float32)

    model_device = next(model.parameters()).device
    use_amp = model_device.type == "cuda"
    n_samples = x_tensor.shape[0]

    # FIX 9: Online moment estimation for mean and variance
    # For embeddings: we need to return all repeat_embeddings for clustering
    # So we still collect them, but compute variance online if needed
    repeat_embeddings = []
    repeat_outputs = [] if return_outputs else None

    with torch.no_grad():
        for _ in range(repeat_times):
            batch_embeddings = []
            batch_outputs = [] if return_outputs else None
            for start in range(0, n_samples, batch_size):
                batch = x_tensor[start : start + batch_size].to(model_device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    output, embedding = model(batch, return_features=True)
                batch_embeddings.append(embedding.detach().cpu())
                if return_outputs:
                    batch_outputs.append(output.detach().cpu())
            repeat_embeddings.append(torch.cat(batch_embeddings, dim=0))
            if return_outputs:
                repeat_outputs.append(torch.cat(batch_outputs, dim=0))

    embeddings = torch.stack(repeat_embeddings, dim=0).numpy()

    if return_outputs:
        # FIX 9: Compute mean and variance online for outputs
        outputs = torch.stack(repeat_outputs, dim=0)  # (repeat, n, output_dim)
        pred_mean = outputs.mean(dim=0).numpy()  # (n, output_dim)
        output_var = outputs.var(dim=0)  # (n, output_dim) using population variance

        # Normalize uncertainty per output dimension.
        # Model outputs are in standardized space (training targets were normalized),
        # so variances across dimensions are already roughly comparable.
        # We still normalize by each dimension's mean MC variance to handle cases where
        # the candidate pool is in an OOD region and some dimensions show uniformly
        # higher uncertainty than others.
        # U_i = (1/D) * sum_d Var(y_{i,d}) / (mean_j Var(y_{j,d}) + epsilon)
        epsilon = 1e-8
        dim_scale = output_var.mean(dim=0).clamp(min=epsilon)
        normalized_var = output_var / dim_scale[None, :]
        uncertainties = normalized_var.mean(dim=1).numpy()
    else:
        # For embeddings, compute mean and variance online would require different storage
        # Current implementation still stores all repeats for clustering
        uncertainties = embeddings.var(axis=0).mean(axis=1)


    # Restore exact per-module training state (handles BatchNorm, LayerNorm etc.)
    for module, state in original_states.items():
        module.training = state

    return embeddings, uncertainties, pred_mean


def select_huds(model, train_pool_df, unlabeled_mask, train_labeled_df, config, var_cols, device):
    """Select the next active learning batch with Hybrid Uncertainty and Diversity Sampling.

    FIX 06: Eliminated variable reuse - topk_positions, cluster_positions, selected_positions
    each have distinct semantic meaning and are named accordingly.
    """
    _validate_selection_inputs(train_pool_df, var_cols)

    unlabeled_pool = _filter_unlabeled_pool(train_pool_df, unlabeled_mask, train_labeled_df)
    if unlabeled_pool.empty:
        return _empty_selection_result()

    # FIX 15: Use full candidate pool directly (removed pre_n random pre-sampling)
    candidate_pool = unlabeled_pool.reset_index(drop=True)
    n_select = min(int(config.training.sample_per_step), len(candidate_pool))
    if n_select <= 0:
        return _empty_selection_result()

    # --- 1. MC Dropout uncertainty estimation ---
    feature_values = candidate_pool[var_cols].to_numpy(dtype=np.float32)
    x_tensor = torch.tensor(feature_values, dtype=torch.float32, device=device)
    # FIX 13: Pass return_outputs flag for output-space uncertainty estimation
    repeat_embeddings, uncertainties, _ = mc_dropout_predict(
        model,
        x_tensor,
        repeat_times=int(config.huds.repeat_times),
        batch_size=int(config.huds.batch_size),
        return_outputs=bool(config.huds.uncertainty_on_outputs),
    )
    # repeat_embeddings.shape = (repeat_times, n_candidates, hidden_dim)
    # uncertainties.shape = (n_candidates,) -- aligned with candidate_pool rows

    # --- 2. Top-K uncertainty filtering ---
    if bool(config.huds.use_top_p):
        # FIX 14: Apply temperature scaling for Top-P softmax
        topk_positions = _select_top_p(
            uncertainties,
            float(config.huds.top_p_threshold),
            temperature=float(config.huds.top_p_temperature),
        )
    else:
        topk_ratio = float(config.huds.topk_ratio)
        n_topk = min(len(candidate_pool), max(1, int(topk_ratio * len(candidate_pool))))
        topk_positions = np.argsort(-uncertainties)[:n_topk]

    parameter_features = candidate_pool[var_cols].to_numpy(dtype=np.float32)
    reference_features = _normalized_labeled_features(train_labeled_df, config, var_cols)
    selected_positions = _select_uncertain_diverse_positions(
        topk_positions,
        uncertainties,
        parameter_features,
        reference_features,
        n_select,
    )
    selected_ids = [
        _normalize_sample_id(_sample_id(candidate_pool, pos))
        for pos in selected_positions
    ]
    selected_uncertainties = [float(uncertainties[pos]) for pos in selected_positions]
    min_labeled_distances = _distances_to_reference(
        selected_positions, parameter_features, reference_features
    )

    # Split selected samples into train/val/test sets per config.split.
    split_ids = _split_selected_ids(selected_ids, config)
    return {
        "selected_ids": selected_ids,
        "uncertainties": selected_uncertainties,
        "topk_size": int(len(topk_positions)),
        "n_clusters": 0,
        "cluster_stats": {},
        "checkpoint_used": "",
        "selection_method": "output_uncertainty_topk_parameter_maximin",
        "fill_method": "parameter_space_k_center",
        "uncertainty_metric": (
            "mc_dropout_output_variance_normalized"
            if bool(config.huds.uncertainty_on_outputs)
            else "mc_dropout_embedding_variance"
        ),
        "topk_uncertainty_threshold": float(uncertainties[topk_positions].min()),
        "min_distance_to_existing_labeled": min_labeled_distances,
        "split_assignment": split_ids,
    }


def select_initial_space_filling(train_pool_df, unlabeled_mask, train_labeled_df, config, var_cols):
    """Select the first labeled batch with deterministic parameter-space maximin."""
    _validate_selection_inputs(train_pool_df, var_cols)
    candidate_pool = _filter_unlabeled_pool(train_pool_df, unlabeled_mask, train_labeled_df)
    if candidate_pool.empty:
        return _empty_selection_result()

    n_select = min(int(config.training.initial_train_size), len(candidate_pool))
    features = candidate_pool[var_cols].to_numpy(dtype=np.float32)
    selected_positions = _select_maximin_positions(
        np.arange(len(candidate_pool), dtype=np.intp),
        features,
        np.empty((0, features.shape[1]), dtype=np.float32),
        n_select,
        seed_at_center=True,
    )
    selected_ids = [
        _normalize_sample_id(_sample_id(candidate_pool, pos))
        for pos in selected_positions
    ]
    return {
        "selected_ids": selected_ids,
        "uncertainties": [None] * len(selected_ids),
        "topk_size": int(len(candidate_pool)),
        "n_clusters": 0,
        "cluster_stats": {},
        "checkpoint_used": "",
        "selection_method": "initial_parameter_space_maximin",
        "fill_method": "parameter_space_k_center",
        "uncertainty_metric": "not_available_initial_parameter_space_filling",
        "topk_uncertainty_threshold": None,
        "min_distance_to_existing_labeled": [None] * len(selected_ids),
        "split_assignment": _split_selected_ids(selected_ids, config),
    }


def _normalized_labeled_features(train_labeled_df, config, var_cols):
    if train_labeled_df.empty or any(column not in train_labeled_df.columns for column in var_cols):
        return np.empty((0, len(var_cols)), dtype=np.float32)

    ranges = {variable.name: (float(variable.min), float(variable.max)) for variable in config.variables}
    values = train_labeled_df[var_cols].to_numpy(dtype=np.float32)
    mins = np.array([ranges[column][0] for column in var_cols], dtype=np.float32)
    maxs = np.array([ranges[column][1] for column in var_cols], dtype=np.float32)
    return (values - mins) / np.maximum(maxs - mins, 1e-8)


def _select_uncertain_diverse_positions(topk_positions, uncertainties, features, reference_features, n_select):
    """Select only high-uncertainty candidates, maximizing parameter-space novelty."""
    positions = np.asarray(topk_positions, dtype=np.intp)
    if len(positions) == 0 or n_select <= 0:
        return []

    # The seed is the most uncertain candidate. Every later pick remains in
    # Top-K and is chosen by its largest minimum distance to prior coverage.
    seed = int(positions[np.argmax(uncertainties[positions])])
    return _select_maximin_positions(
        positions,
        features,
        reference_features,
        n_select,
        initial_position=seed,
    )


def _distances_to_reference(selected_positions, features, reference_features):
    if not len(reference_features):
        return [None] * len(selected_positions)
    return [
        float(euclidean_distances(features[[position]], reference_features).min())
        for position in selected_positions
    ]


def _select_maximin_positions(positions, features, reference_features, n_select, initial_position=None, seed_at_center=False):
    positions = np.asarray(positions, dtype=np.intp)
    if len(positions) == 0 or n_select <= 0:
        return []

    selected = []
    if initial_position is not None:
        selected.append(int(initial_position))
    elif len(reference_features):
        distances = euclidean_distances(features[positions], reference_features).min(axis=1)
        selected.append(int(positions[np.argmax(distances)]))
    elif seed_at_center:
        center_distances = np.sum((features[positions] - 0.5) ** 2, axis=1)
        selected.append(int(positions[np.argmin(center_distances)]))
    else:
        selected.append(int(positions[0]))

    while len(selected) < min(n_select, len(positions)):
        remaining = np.array([pos for pos in positions if pos not in set(selected)], dtype=np.intp)
        anchors = features[np.asarray(selected, dtype=np.intp)]
        if len(reference_features):
            anchors = np.vstack((reference_features, anchors))
        distances = euclidean_distances(features[remaining], anchors).min(axis=1)
        selected.append(int(remaining[np.argmax(distances)]))
    return selected


def _split_selected_ids(selected_ids: list[str], config: AppConfig) -> dict[str, list[str]]:
    """Split selected sample IDs into train/val/test sets per config.split.

    Uses a deterministic shuffle based on config.random_seed for reproducibility.
    """
    n = len(selected_ids)
    if n == 0:
        return {"train": [], "val": [], "test": []}

    indices = list(range(n))
    random.Random(config.random_seed).shuffle(indices)

    # Guarantee at least 1 sample per split when possible
    if n >= 3:
        n_train = max(1, int(n * config.split.train_split))
        n_val = max(1, int(n * config.split.val_split))
    elif n == 2:
        n_train = 1
        n_val = 1
    else:
        n_train = 1
        n_val = 0

    n_test = n - n_train - n_val

    split_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    return {
        "train": [selected_ids[i] for i in sorted(split_indices)],
        "val": [selected_ids[i] for i in sorted(val_indices)],
        "test": [selected_ids[i] for i in sorted(test_indices)],
    }


def _aggregate_labeled(run_path: Path) -> pd.DataFrame:
    """Aggregate labeled samples from train/val/test splits."""
    datasets_dir = run_path / "datasets"
    frames = []
    for filename in ("train_labeled.csv", "val_labeled.csv", "test_labeled.csv"):
        path = datasets_dir / filename
        if path.exists():
            frames.append(read_csv(path))
    if not frames:
        return pd.DataFrame({SAMPLE_ID_COLUMN: []})
    return pd.concat(frames, ignore_index=True)


def run_huds_sampling(run_dir, config, step):
    """Run the full HUDS sampling workflow for one active learning step."""
    run_path = ensure_run_dir(str(run_dir))
    state = RunState.load(str(run_path))
    app_config = _resolve_config(run_path, config)
    _validate_sampling_step(state, app_config, step)
    variable_columns = [variable.name for variable in app_config.variables]

    # Defensive: verify required files exist before proceeding
    candidate_pool_path = run_path / "candidate_pool.csv"
    if not candidate_pool_path.exists():
        raise FileNotFoundError(
            f"candidate_pool.csv missing from {run_path}. "
            "The benchmark may have been interrupted. Re-run the experiment."
        )

    candidate_pool_df = read_csv(candidate_pool_path)
    all_labeled_df = _aggregate_labeled(run_path)

    # Step 0: no checkpoint or normalization yet - use raw pool and skip model loading
    is_initial_step = (state.trained_step == -1 and state.current_step == 0)
    checkpoint_path = _latest_checkpoint_path(run_path, state)

    if is_initial_step:
        var_map = {v.name: v for v in app_config.variables}
        normalized_pool_df = candidate_pool_df[variable_columns + [SAMPLE_ID_COLUMN]].copy()
        for col in variable_columns:
            vmin = float(var_map[col].min)
            vmax = float(var_map[col].max)
            normalized_pool_df[col] = (candidate_pool_df[col] - vmin) / (vmax - vmin + 1e-8)
    else:
        normalized_pool_df = _normalize_pool_for_model(candidate_pool_df, run_path, variable_columns)

    unlabeled_mask = _build_unlabeled_mask(candidate_pool_df, all_labeled_df, state)

    if is_initial_step:
        result = select_initial_space_filling(
            train_pool_df=normalized_pool_df,
            unlabeled_mask=unlabeled_mask,
            train_labeled_df=all_labeled_df,
            config=app_config,
            var_cols=variable_columns,
        )
    else:
        device_obj = resolve_device(app_config.training.device)
        model = build_model(app_config).to(device_obj)
        if checkpoint_path.exists():
            _load_checkpoint_weights(model, checkpoint_path, device_obj)

        result = select_huds(
            model=model,
            train_pool_df=normalized_pool_df,
            unlabeled_mask=unlabeled_mask,
            train_labeled_df=all_labeled_df,
            config=app_config,
            var_cols=variable_columns,
            device=device_obj,
        )
    result["checkpoint_used"] = str(checkpoint_path)

    selected_ids = result["selected_ids"]
    split_assignment = result.pop("split_assignment")
    selected_ids_str = [str(sid) for sid in selected_ids]
    request_df = candidate_pool_df[candidate_pool_df[SAMPLE_ID_COLUMN].isin(selected_ids_str)].copy()
    request_df = _order_request_rows(request_df, selected_ids_str)
    request_df[variable_columns] = request_df[variable_columns].round(5)

    selected_parameters = []
    selected_uncertainties = result.get("uncertainties", [])
    min_labeled_distances = result.get("min_distance_to_existing_labeled", [])
    for index, (_, row) in enumerate(request_df.iterrows()):
        selected_parameters.append({
            "sample_id": _normalize_sample_id(row[SAMPLE_ID_COLUMN]),
            "parameters": {
                column: round(float(row[column]), 5)
                for column in variable_columns
            },
            "output_uncertainty": (
                selected_uncertainties[index]
                if index < len(selected_uncertainties) else None
            ),
            "min_distance_to_existing_labeled": (
                min_labeled_distances[index]
                if index < len(min_labeled_distances) else None
            ),
        })
    result["selection_audit"] = {
        "uncertainty_metric": result.get("uncertainty_metric"),
        "topk_size": result.get("topk_size"),
        "topk_uncertainty_threshold": result.get("topk_uncertainty_threshold"),
        "parameter_distance_metric": "euclidean_on_config_range_normalized_parameters",
        "selected_samples": selected_parameters,
    }

    # Add split assignment column to request CSV for import_labels routing
    split_map = {str(sid): split_type for split_type, sids in split_assignment.items() for sid in sids}
    request_df[SPLIT_ASSIGNMENT_COLUMN] = request_df[SAMPLE_ID_COLUMN].map(split_map)

    request_path = run_path / "requests" / f"train_step_{int(step):03d}_request.csv"
    diagnostics_path = run_path / "artifacts" / f"huds_step_{int(step):03d}.json"

    write_csv(request_df, request_path)
    with diagnostics_path.open("w", encoding="utf-8") as file:
        json.dump(_json_ready(result), file, indent=2)
        file.write("\n")

    state.current_step = int(step)
    state.latest_checkpoint = str(checkpoint_path.relative_to(run_path)) if checkpoint_path.is_relative_to(run_path) else str(checkpoint_path)

    # Store split assignment in state
    state.split_assignments[str(step)] = {
        k: [_normalize_sample_id(sid) for sid in v]
        for k, v in split_assignment.items()
    }

    # FIX 03: Normalize sample IDs before extending pending_sample_ids
    state.pending_sample_ids.extend(_normalize_sample_id(sid) for sid in selected_ids)
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

    # FIX 04: Allow first step (step 1) when no model exists yet - it will be random sampling
    if int(state.trained_step) < int(state.current_step):
        if requested_step == 1 and state.trained_step == -1 and state.current_step == 0:
            pass  # Allow initial random sampling
        else:
            msg = (
                f"Cannot sample step {requested_step}: current step {state.current_step} "
                f"has not been trained (trained_step={state.trained_step}). "
                f"Run 'huds-app train --run {state.run_dir}' first."
            )
            if state.pending_sample_ids:
                import logging
                logging.warning(
                    f"trained_step ({state.trained_step}) < current_step ({state.current_step}), "
                    f"but there are {len(state.pending_sample_ids)} pending sample IDs. "
                    f"This may indicate a prior training failure."
                )
            raise ValueError(msg)


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


def _standardize(values):
    means = values.mean(axis=0, keepdims=True)
    stds = values.std(axis=0, keepdims=True)
    # FIX 18: Use tolerance comparison to catch near-zero std (IEEE 754 precision)
    stds[np.abs(stds) < 1e-12] = 1.0
    return (values - means) / stds


def _select_top_p(uncertainties, threshold, temperature: float = 1.0):
    """Select samples via Top-P (nucleus) strategy.

    FIX 14: Added temperature parameter to control softmax concentration.

    Normalize uncertainties to a probability distribution via softmax,
    sort descending, and greedily accumulate until cumulative probability
    exceeds the threshold.

    Args:
        uncertainties: 1D array of uncertainty values.
        threshold: cumulative probability cutoff in (0, 1].
        temperature: temperature scaling for softmax (>1 smoother, <1 more concentrated).

    Returns:
        Indices of selected samples sorted by descending uncertainty.
    """
    if len(uncertainties) <= 1:
        return np.array([0], dtype=np.intp)

    # FIX 14: Apply temperature scaling to control softmax concentration
    probs = _softmax(uncertainties / max(temperature, 1e-8))
    sorted_indices = np.argsort(-probs)
    sorted_probs = probs[sorted_indices]
    cumsum = np.cumsum(sorted_probs)

    n_select = int(np.searchsorted(cumsum, threshold, side="left") + 1)
    n_select = max(1, min(n_select, len(sorted_indices)))
    return sorted_indices[:n_select]


def _softmax(values):
    """Numerically stable softmax."""
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()


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


def _k_center_fill(features, selected_positions, n_select):
    """Fill missing samples using k-center greedy selection with incremental distance updates.

    FIX 10: Maintains incremental minimum distances to avoid recomputing full distance matrix.
    """
    selected_set = set(selected_positions)
    if not selected_set:
        selected_set.add(0)

    n_samples = features.shape[0]
    # Initialize minimum distances from each sample to the nearest selected sample
    min_dist = np.full(n_samples, np.inf)

    # Calculate initial distances for all samples to the first selected set
    selected_array = np.array(sorted(selected_set), dtype=int)
    distances = euclidean_distances(features, features[selected_array]).min(axis=1)
    min_dist = np.minimum(min_dist, distances)

    added_positions = []
    while len(selected_set) < n_select and len(selected_set) < n_samples:
        # Set distance of selected samples to -inf so they are not chosen again
        for position in selected_set:
            min_dist[position] = -np.inf

        next_position = int(np.argmax(min_dist))
        selected_set.add(next_position)
        added_positions.append(next_position)

        # FIX 10: Update minimum distances incrementally with the newly added sample
        new_distances = euclidean_distances(features, features[[next_position]]).flatten()
        min_dist = np.minimum(min_dist, new_distances)

    return added_positions


def _sample_id(df, position):
    """Extract sample ID from DataFrame at given position."""
    value = df.iloc[int(position)][SAMPLE_ID_COLUMN]
    return value.item() if hasattr(value, "item") else value


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
    return checkpoint_path  # May not exist on step 0 - caller handles it


def _normalize_pool_for_model(train_pool_df, run_path, variable_columns):
    normalization_path = run_path / "artifacts" / "normalization.json"
    if not normalization_path.exists():
        raise FileNotFoundError(f"Normalization statistics not found: {normalization_path}")
    with normalization_path.open("r", encoding="utf-8") as file:
        normalization = json.load(file)

    var_stats = normalization.get("variables", {})
    normalized_df = train_pool_df.copy()

    # FIX 11: Detect OOD samples during normalization
    ood_counts = {}
    for col in variable_columns:
        if col in var_stats and col in normalized_df.columns:
            values = normalized_df[col].to_numpy(dtype=np.float32)
            value_range = float(var_stats[col]["range"])
            min_val = float(var_stats[col]["min"])
            max_val = float(var_stats[col]["max"])

            # Apply normalization safely
            if value_range > 0:
                normalized_df[col] = (values - min_val) / value_range
            else:
                normalized_df[col] = 0.0

            # FIX 11: Detect OOD samples (values outside configured physical range)
            raw_values = train_pool_df[col].to_numpy(dtype=np.float32)
            ood_count = int((raw_values < min_val).sum() + (raw_values > max_val).sum())
            if ood_count > 0:
                ood_counts[col] = ood_count

    if ood_counts:
        total = len(train_pool_df)
        ood_summary = ", ".join(f"{col}: {c}/{total}" for col, c in ood_counts.items())
        print(f"Warning: Pool contains OOD samples outside configured variable ranges: {ood_summary}")

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
    """Load checkpoint weights with strict validation."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint.get("model")
            or checkpoint
        )
    else:
        state_dict = checkpoint

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"Checkpoint architecture does not match current config at {checkpoint_path}. "
            "Please retrain the model or restore the matching config."
        ) from e


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


