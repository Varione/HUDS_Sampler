import numpy as np


def r2_score(y_true, y_pred):
    if len(y_true) < 2:
        return float("nan")
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mean_predictive_variance(pred_mean, pred_var):
    return float(np.mean(pred_var))


def compute_metrics(y_true, y_pred, output_names=None):
    n_outputs = y_true.shape[1] if y_true.ndim > 1 else 1
    metrics = {}
    for i in range(n_outputs):
        name = output_names[i] if output_names else f"output_{i}"
        metrics[f"r2_{name}"] = r2_score(y_true[:, i], y_pred[:, i])
        metrics[f"rmse_{name}"] = rmse(y_true[:, i], y_pred[:, i])

    r2_vals = [metrics.get(f"r2_{n}", 0) for n in (output_names or [f"output_{i}" for i in range(n_outputs)])]
    valid_r2 = [v for v in r2_vals if not np.isnan(v)]
    metrics["r2_avg"] = float(np.mean(valid_r2)) if valid_r2 else float("nan")
    return metrics
