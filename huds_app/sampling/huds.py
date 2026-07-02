from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

from huds_app.core.config import AppConfig, load_config
from huds_app.data.schema import SAMPLE_ID_COLUMN, STATUS_COLUMN
from huds_app.model.architecture import build_model
from huds_app.core.storage import RunState, _normalize_sample_id, ensure_run_dir, read_csv, write_csv, resolve_device


def mc_dropout_predict(model, x, repeat_times, batch_size, return_outputs: bool = False):
    """Run MC Dropout and return repeat embeddings plus variance uncertainty.

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
    use_amp = model_device.type == "cuda"
    n_samples = x_tensor.shape[0]
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
        outputs = torch.stack(repeat_outputs, dim=0)  # (repeat, n, output_dim)
        output_var = outputs.var(dim=0)  # (n, output_dim)
        uncertainties = output_var.sum(dim=1).numpy()  # (n,) total predictive variance
        pred_mean = outputs.mean(dim=0).numpy()  # (n, output_dim)
    else:
        uncertainties = embeddings.var(axis=0).mean(axis=1)
        pred_mean = None

    if not was_training:
        model.eval()

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

    topk_pool = candidate_pool.iloc[topk_positions].copy()
    topk_uncertainties = uncertainties[topk_positions]
    n_clusters = min(n_select, len(topk_positions))

    # --- 3. Embedding space clustering ---
    mean_embedding = repeat_embeddings.mean(axis=0)  # (n_candidates, hidden_dim)
    standardized_topk = _standardize(mean_embedding[topk_positions].astype(np.float32))
    labels = _cluster_topk(standardized_topk, n_clusters, config)

    # --- 4. Cluster representative selection (returns positions in candidate_pool) ---
    cluster_positions: list[int] = []
    cluster_stats_out: dict[str, dict] = {}
    for cluster_id in range(n_clusters):
        member_indices = np.flatnonzero(labels == cluster_id)  # local indices in topk_pool

        # FIX 16: Skip clusters with fewer than 2 members (no statistical significance)
        if member_indices.size < 2:
            cluster_stats_out[str(cluster_id)] = {
                "size": int(member_indices.size), "selected_id": None,
                "max_uncertainty": None, "mean_uncertainty": None,
                "skipped": True,
            }
            continue

        if member_indices.size == 0:
            cluster_stats_out[str(cluster_id)] = {
                "size": 0, "selected_id": None,
                "max_uncertainty": None, "mean_uncertainty": None,
            }
            continue

        # Pick the member with highest uncertainty within this cluster
        local_best = member_indices[np.argmax(topk_uncertainties[member_indices])]
        candidate_position = int(topk_pool.index[local_best])  # position in candidate_pool

        cluster_positions.append(candidate_position)
        cluster_uncertainties = topk_uncertainties[member_indices]
        cluster_stats_out[str(cluster_id)] = {
            "size": int(member_indices.size),
            "selected_id": _normalize_sample_id(_sample_id(topk_pool, local_best)),
            "max_uncertainty": float(cluster_uncertainties.max()),
            "mean_uncertainty": float(cluster_uncertainties.mean()),
        }

    # --- 5. Build result ---
    if not cluster_positions:
        # FIX 21: All clusters skipped (<2 members each), fall back to Top-K direct selection
        fallback_n = min(n_select, len(candidate_pool))
        fallback_indices = np.argsort(-uncertainties)[:fallback_n]
        selected_ids = [
            _normalize_sample_id(_sample_id(candidate_pool, int(pos)))
            for pos in fallback_indices
        ]
        selected_uncertainties = [float(uncertainties[pos]) for pos in fallback_indices]
        cluster_positions = [int(pos) for pos in fallback_indices]
        print(
            f"Warning: all {n_clusters} clusters had <2 members, "
            f"falling back to Top-K selection ({len(fallback_indices)} samples)"
        )
    else:
        selected_ids = [
            _normalize_sample_id(_sample_id(candidate_pool, pos))
            for pos in cluster_positions
        ]
        selected_uncertainties = [float(uncertainties[pos]) for pos in cluster_positions]

    # --- 6. Fill with k-center if needed ---
    if len(selected_ids) < n_select:
        standardized_all = _standardize(mean_embedding.astype(np.float32))
        fill_positions = _k_center_fill(
            standardized_all,
            cluster_positions,
            n_select,
        )
        for pos in fill_positions:
            cluster_positions.append(pos)
            selected_ids.append(_normalize_sample_id(_sample_id(candidate_pool, pos)))
            selected_uncertainties.append(float(uncertainties[pos]))

    return {
        "selected_ids": selected_ids,
        "uncertainties": selected_uncertainties,
        "topk_size": int(len(cluster_positions)),
        "n_clusters": int(n_clusters),
        "cluster_stats": cluster_stats_out,
        "checkpoint_used": "",
    }


def run_huds_sampling(run_dir, config, step):
    """Run the full HUDS sampling workflow for one active learning step."""
    run_path = ensure_run_dir(str(run_dir))
    state = RunState.load(str(run_path))
    app_config = _resolve_config(run_path, config)
    _validate_sampling_step(state, app_config, step)
    variable_columns = [variable.name for variable in app_config.variables]

    # Defensive: verify required files exist before proceeding
    train_pool_path = run_path / "train_pool.csv"
    if not train_pool_path.exists():
        raise FileNotFoundError(
            f"train_pool.csv missing from {run_path}. "
            "The benchmark may have been interrupted. Re-run the experiment."
        )

    train_pool_df = read_csv(train_pool_path)
    train_labeled_path = run_path / "datasets" / "train_labeled.csv"
    train_labeled_df = read_csv(train_labeled_path) if train_labeled_path.exists() else pd.DataFrame({SAMPLE_ID_COLUMN: []})

    checkpoint_path = _latest_checkpoint_path(run_path, state)
    normalized_pool_df = _normalize_pool_for_model(train_pool_df, run_path, variable_columns)
    unlabeled_mask = _build_unlabeled_mask(train_pool_df, train_labeled_df, state)

    device_obj = resolve_device(app_config.training.device)
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

    # FIX 04: Enhanced trained_step validation with better diagnostics
    if int(state.trained_step) < int(state.current_step):
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
    """Load checkpoint weights with strict fallback.

    FIX 07: Try strict=True first, fall back to strict=False with diagnostic logging.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
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
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Warning: strict checkpoint load failed ({e}), attempting non-strict load...")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        if len(missing) > len(model.state_dict()) * 0.5:
            print("Warning: >50% of model keys missing from checkpoint, model may not be properly initialized")


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


