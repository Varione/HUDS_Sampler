import numpy as np


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    residual_sum_squares = np.sum((y_true - y_pred) ** 2)
    total_sum_squares = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)

    if total_sum_squares == 0:
        return 0.0

    return float(1.0 - residual_sum_squares / total_sum_squares)


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_metrics(y_true, y_pred, output_names):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    metrics = {}
    r2_values = []

    for output_index, output_name in enumerate(output_names):
        output_true = y_true[:, output_index]
        output_pred = y_pred[:, output_index]

        output_r2 = r2_score(output_true, output_pred)
        metrics[f"r2_{output_name}"] = output_r2
        metrics[f"rmse_{output_name}"] = rmse(output_true, output_pred)
        r2_values.append(output_r2)

    metrics["r2_avg"] = float(np.mean(r2_values)) if r2_values else 0.0
    return metrics


def mean_predictive_variance(predictions):
    predictions = np.asarray(predictions, dtype=float)

    return np.mean(np.var(predictions, axis=0), axis=1)
