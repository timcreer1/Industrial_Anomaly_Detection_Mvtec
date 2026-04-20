"""
Threshold utilities for the MVTec industrial anomaly detection project.

This module keeps the threshold logic separate from model scoring so the same
policy definitions can be reused across ImageNet, SSL, and reconstruction
methods. The helpers are designed around the project workflow where thresholds
are selected from validation-good image scores and then applied to the test set.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics_utils import (
    compute_threshold_metrics,
    summarise_threshold_metrics,
    summarise_threshold_metrics_by_category,
)


#------------------------------------------------------------------------------
# Default threshold policy settings
#------------------------------------------------------------------------------

DEFAULT_POLICY_QUANTILES = {
    "high_recall": 0.90,
    "balanced": 0.95,
    "low_false_alarm": 0.99,
}


#------------------------------------------------------------------------------
# Basic score cleaning helpers
#------------------------------------------------------------------------------

# Clean a score array into a 1D finite float numpy array
def clean_score_array(scores):
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return arr


# Convert a pandas column or python list into a clean numpy array
def get_clean_scores_from_series(values):
    return clean_score_array(values)


#------------------------------------------------------------------------------
# Validation threshold selection
#------------------------------------------------------------------------------

# Compute one quantile-based threshold from validation-good scores
def compute_quantile_threshold(val_scores, quantile):
    scores = clean_score_array(val_scores)

    if scores.size == 0:
        raise ValueError("No finite validation scores were provided.")

    q = float(np.clip(quantile, 0.0, 1.0))
    return float(np.quantile(scores, q))


# Compute all named policy thresholds from validation-good scores
def compute_policy_thresholds(val_scores, policy_quantiles=None):
    policy_quantiles = policy_quantiles or DEFAULT_POLICY_QUANTILES

    scores = clean_score_array(val_scores)

    if scores.size == 0:
        raise ValueError("No finite validation scores were provided.")

    rows = []
    thresholds = {}

    for policy_name, quantile in policy_quantiles.items():
        threshold = compute_quantile_threshold(scores, quantile)
        thresholds[policy_name] = threshold

        rows.append({
            "policy": str(policy_name),
            "quantile": float(quantile),
            "threshold": float(threshold),
            "n_val_scores": int(scores.size),
            "val_score_min": float(np.min(scores)),
            "val_score_mean": float(np.mean(scores)),
            "val_score_max": float(np.max(scores)),
        })

    threshold_df = pd.DataFrame(rows).sort_values("quantile").reset_index(drop=True)
    return thresholds, threshold_df


# Build a compact row table describing validation score coverage
def build_val_score_summary_table(val_scores, label="validation_good"):
    scores = clean_score_array(val_scores)

    if scores.size == 0:
        return pd.DataFrame([{
            "label": label,
            "n_scores": 0,
            "score_min": np.nan,
            "score_p25": np.nan,
            "score_median": np.nan,
            "score_p75": np.nan,
            "score_p90": np.nan,
            "score_p95": np.nan,
            "score_p99": np.nan,
            "score_max": np.nan,
            "score_mean": np.nan,
            "score_std": np.nan,
        }])

    return pd.DataFrame([{
        "label": label,
        "n_scores": int(scores.size),
        "score_min": float(np.min(scores)),
        "score_p25": float(np.quantile(scores, 0.25)),
        "score_median": float(np.quantile(scores, 0.50)),
        "score_p75": float(np.quantile(scores, 0.75)),
        "score_p90": float(np.quantile(scores, 0.90)),
        "score_p95": float(np.quantile(scores, 0.95)),
        "score_p99": float(np.quantile(scores, 0.99)),
        "score_max": float(np.max(scores)),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
    }])


#------------------------------------------------------------------------------
# Prediction helpers
#------------------------------------------------------------------------------

# Convert anomaly scores into binary predictions using one threshold
def predict_from_threshold(scores, threshold):
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    preds = (arr >= float(threshold)).astype(int)
    return preds


# Add prediction columns for one or more threshold policies to an eval table
def add_policy_prediction_columns(
    eval_df,
    thresholds,
    score_col="image_score",
    prefix="pred",
):
    if score_col not in eval_df.columns:
        raise KeyError(f"'{score_col}' is not present in eval_df.")

    out_df = eval_df.copy()

    for policy_name, threshold in thresholds.items():
        pred_col = f"{prefix}_{policy_name}"
        out_df[pred_col] = (out_df[score_col].astype(float) >= float(threshold)).astype(int)

    return out_df


# Add both threshold and prediction columns to an eval table
def add_policy_columns(
    eval_df,
    thresholds,
    score_col="image_score",
    threshold_prefix="threshold",
    pred_prefix="pred",
):
    if score_col not in eval_df.columns:
        raise KeyError(f"'{score_col}' is not present in eval_df.")

    out_df = eval_df.copy()

    for policy_name, threshold in thresholds.items():
        threshold_col = f"{threshold_prefix}_{policy_name}"
        pred_col = f"{pred_prefix}_{policy_name}"

        out_df[threshold_col] = float(threshold)
        out_df[pred_col] = (out_df[score_col].astype(float) >= float(threshold)).astype(int)

    return out_df


#------------------------------------------------------------------------------
# Policy evaluation helpers
#------------------------------------------------------------------------------

# Evaluate one threshold policy on a full eval dataframe
def evaluate_single_policy(
    eval_df,
    threshold,
    score_col="image_score",
    label_col="is_anomaly",
):
    if score_col not in eval_df.columns:
        raise KeyError(f"'{score_col}' is not present in eval_df.")
    if label_col not in eval_df.columns:
        raise KeyError(f"'{label_col}' is not present in eval_df.")

    y_true = eval_df[label_col].astype(int).to_numpy()
    scores = eval_df[score_col].astype(float).to_numpy()
    y_pred = predict_from_threshold(scores, threshold)

    return compute_threshold_metrics(y_true, y_pred)


# Evaluate named threshold policies on one eval dataframe
def evaluate_policy_thresholds(
    eval_df,
    thresholds,
    score_col="image_score",
    label_col="is_anomaly",
    category_col="category",
    model_name=None,
):
    if score_col not in eval_df.columns:
        raise KeyError(f"'{score_col}' is not present in eval_df.")
    if label_col not in eval_df.columns:
        raise KeyError(f"'{label_col}' is not present in eval_df.")
    if category_col not in eval_df.columns:
        raise KeyError(f"'{category_col}' is not present in eval_df.")

    full_rows = []
    category_rows = []
    mean_rows = []

    for policy_name, threshold in thresholds.items():
        df_policy = eval_df.copy()
        df_policy["policy"] = str(policy_name)
        df_policy["threshold"] = float(threshold)
        df_policy["y_pred"] = predict_from_threshold(df_policy[score_col].to_numpy(), threshold)

        overall_metrics = evaluate_single_policy(
            df_policy,
            threshold=threshold,
            score_col=score_col,
            label_col=label_col,
        )
        overall_metrics["policy"] = str(policy_name)
        overall_metrics["threshold"] = float(threshold)
        overall_metrics["n_samples"] = int(len(df_policy))
        if model_name is not None:
            overall_metrics["model"] = str(model_name)
        full_rows.append(overall_metrics)

        category_summary = summarise_threshold_metrics_by_category(
            df_policy.rename(columns={"y_pred": "pred_label"}),
            category_col=category_col,
            label_col=label_col,
            pred_col="pred_label",
        ).copy()
        category_summary["policy"] = str(policy_name)
        category_summary["threshold"] = float(threshold)
        if model_name is not None:
            category_summary["model"] = str(model_name)
        category_rows.append(category_summary)

        mean_summary = summarise_threshold_metrics(category_summary).copy()
        mean_summary["policy"] = str(policy_name)
        mean_summary["threshold"] = float(threshold)
        if model_name is not None:
            mean_summary["model"] = str(model_name)
        mean_rows.append(mean_summary)

    full_df = pd.DataFrame(full_rows)
    category_df = pd.concat(category_rows, axis=0, ignore_index=True) if category_rows else pd.DataFrame()
    mean_df = pd.concat(mean_rows, axis=0, ignore_index=True) if mean_rows else pd.DataFrame()

    return full_df, category_df, mean_df


# Build a compact threshold-policy summary table for reporting
def build_policy_summary_table(
    threshold_full_df,
    metric_cols=None,
):
    metric_cols = metric_cols or [
        "precision",
        "recall",
        "f1",
        "accuracy",
        "fpr",
        "tpr",
        "tp",
        "tn",
        "fp",
        "fn",
    ]

    keep_cols = ["model", "policy", "threshold"]
    keep_cols = [c for c in keep_cols if c in threshold_full_df.columns]
    keep_cols += [c for c in metric_cols if c in threshold_full_df.columns]

    out_df = threshold_full_df[keep_cols].copy()

    sort_cols = [c for c in ["model", "policy"] if c in out_df.columns]
    if sort_cols:
        out_df = out_df.sort_values(sort_cols).reset_index(drop=True)

    return out_df


#------------------------------------------------------------------------------
# Threshold sweep helpers
#------------------------------------------------------------------------------

# Build sweep thresholds from validation-good score quantiles
def build_threshold_sweep_from_val_scores(
    val_scores,
    q_min=0.80,
    q_max=0.999,
    n_steps=40,
):
    scores = clean_score_array(val_scores)

    if scores.size == 0:
        raise ValueError("No finite validation scores were provided.")

    q_min = float(np.clip(q_min, 0.0, 1.0))
    q_max = float(np.clip(q_max, 0.0, 1.0))

    if q_max < q_min:
        raise ValueError("q_max must be >= q_min.")

    quantiles = np.linspace(q_min, q_max, int(n_steps))
    thresholds = [float(np.quantile(scores, q)) for q in quantiles]

    sweep_df = pd.DataFrame({
        "sweep_index": np.arange(len(quantiles)),
        "quantile": quantiles,
        "threshold": thresholds,
    })

    return thresholds, sweep_df


# Evaluate a dense threshold sweep on one eval dataframe
def evaluate_threshold_sweep(
    eval_df,
    thresholds,
    score_col="image_score",
    label_col="is_anomaly",
    category_col="category",
    model_name=None,
):
    if score_col not in eval_df.columns:
        raise KeyError(f"'{score_col}' is not present in eval_df.")
    if label_col not in eval_df.columns:
        raise KeyError(f"'{label_col}' is not present in eval_df.")
    if category_col not in eval_df.columns:
        raise KeyError(f"'{category_col}' is not present in eval_df.")

    full_rows = []
    category_rows = []
    mean_rows = []

    for sweep_index, threshold in enumerate(thresholds):
        df_sweep = eval_df.copy()
        df_sweep["sweep_index"] = int(sweep_index)
        df_sweep["threshold"] = float(threshold)
        df_sweep["y_pred"] = predict_from_threshold(df_sweep[score_col].to_numpy(), threshold)

        overall_metrics = evaluate_single_policy(
            df_sweep,
            threshold=threshold,
            score_col=score_col,
            label_col=label_col,
        )
        overall_metrics["sweep_index"] = int(sweep_index)
        overall_metrics["threshold"] = float(threshold)
        overall_metrics["n_samples"] = int(len(df_sweep))
        if model_name is not None:
            overall_metrics["model"] = str(model_name)
        full_rows.append(overall_metrics)

        category_summary = summarise_threshold_metrics_by_category(
            df_sweep.rename(columns={"y_pred": "pred_label"}),
            category_col=category_col,
            label_col=label_col,
            pred_col="pred_label",
        ).copy()
        category_summary["sweep_index"] = int(sweep_index)
        category_summary["threshold"] = float(threshold)
        if model_name is not None:
            category_summary["model"] = str(model_name)
        category_rows.append(category_summary)

        mean_summary = summarise_threshold_metrics(category_summary).copy()
        mean_summary["sweep_index"] = int(sweep_index)
        mean_summary["threshold"] = float(threshold)
        if model_name is not None:
            mean_summary["model"] = str(model_name)
        mean_rows.append(mean_summary)

    full_df = pd.DataFrame(full_rows)
    category_df = pd.concat(category_rows, axis=0, ignore_index=True) if category_rows else pd.DataFrame()
    mean_df = pd.concat(mean_rows, axis=0, ignore_index=True) if mean_rows else pd.DataFrame()

    return full_df, category_df, mean_df


# Select the best threshold from a sweep table using one metric
def select_best_threshold_from_sweep(
    sweep_df,
    metric="f1",
    maximise=True,
):
    if metric not in sweep_df.columns:
        raise KeyError(f"'{metric}' is not present in sweep_df.")

    df = sweep_df.copy()
    df = df[np.isfinite(df[metric].to_numpy())].copy()

    if df.empty:
        raise ValueError(f"No finite values were found for sweep metric '{metric}'.")

    ascending = not bool(maximise)
    best_idx = df[metric].astype(float).sort_values(ascending=ascending).index[0]
    best_row = df.loc[best_idx].copy()

    return best_row


#------------------------------------------------------------------------------
# Model target selection helpers
#------------------------------------------------------------------------------

# Select the best overall and best SSL models from a main comparison table
def select_threshold_targets(
    main_comparison_mean_df,
    metric_col="image_pr_auc",
    model_col="model",
):
    if metric_col not in main_comparison_mean_df.columns:
        raise KeyError(f"'{metric_col}' is not present in main comparison table.")
    if model_col not in main_comparison_mean_df.columns:
        raise KeyError(f"'{model_col}' is not present in main comparison table.")

    df = main_comparison_mean_df.copy()
    df = df[np.isfinite(df[metric_col].to_numpy())].copy()

    if df.empty:
        raise ValueError("Main comparison table is empty after filtering finite metric rows.")

    df = df.sort_values(metric_col, ascending=False).reset_index(drop=True)

    best_overall = str(df.iloc[0][model_col])

    ssl_mask = df[model_col].astype(str).str.contains("simclr", case=False, na=False)
    if ssl_mask.any():
        best_ssl = str(df.loc[ssl_mask].iloc[0][model_col])
    else:
        best_ssl = None

    return {
        "metric_used": metric_col,
        "best_overall_model": best_overall,
        "best_ssl_model": best_ssl,
    }


#------------------------------------------------------------------------------
# Save and load helpers
#------------------------------------------------------------------------------

# Save a thresholds dictionary to JSON
def save_thresholds_json(thresholds, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    payload = {str(k): float(v) for k, v in thresholds.items()}

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return filepath


# Load a thresholds dictionary from JSON
def load_thresholds_json(filepath):
    filepath = Path(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return {str(k): float(v) for k, v in payload.items()}


# Save a threshold target selection dictionary to JSON
def save_threshold_targets_json(targets, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(targets, f, indent=2)

    return filepath


# Load a threshold target selection dictionary from JSON
def load_threshold_targets_json(filepath):
    filepath = Path(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return payload


# Save a dataframe to CSV
def save_threshold_csv(df, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    return filepath


# Load a CSV dataframe
def load_threshold_csv(filepath):
    return pd.read_csv(filepath)
