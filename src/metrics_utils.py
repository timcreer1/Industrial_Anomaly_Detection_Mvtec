"""Metrics helpers for the MVTec anomaly detection project.

This module keeps together the project-level evaluation helpers used across the
baseline, SSL, threshold, and failure-analysis notebooks. It focuses on:
- safe image-level ROC-AUC and PR-AUC helpers
- pixel-level ROC-AUC from masks and anomaly heatmaps
- threshold metrics for deployment-style operating points
- compact per-image evaluation table builders
- category-level and mean summary table builders
- lightweight save helpers for CSV and JSON artefacts

The goal is to keep the scoring and summarisation logic consistent across the
project while staying notebook-friendly and easy to debug.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


#------------------------------------------------------------------------------
# Small general helpers
#------------------------------------------------------------------------------

# Convert one object into a Path and create the parent directory when needed.
def ensure_parent_dir(file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


# Convert input to a flat NumPy float array.
def as_float_1d(values):
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return arr


# Convert input to a flat NumPy integer array.
def as_int_1d(values):
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    return arr


# Convert an object to a list while handling None cleanly.
def as_list(values):
    if values is None:
        return []
    if isinstance(values, list):
        return values
    return list(values)


# Return a JSON-safe scalar value.
def json_safe_scalar(value):
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


# Return the mean of one array while handling all-NaN inputs.
def safe_nanmean(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.nan
    if np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmean(values))


# Validate that the main image-level arrays align in length.
def validate_basic_lengths(y_true, y_score, categories=None, image_paths=None):
    n_true = len(as_int_1d(y_true))
    n_score = len(as_float_1d(y_score))

    if n_true != n_score:
        raise ValueError("y_true and y_score must have the same length.")

    if categories is not None and len(categories) != n_true:
        raise ValueError("categories must have the same length as y_true.")

    if image_paths is not None and len(image_paths) != n_true:
        raise ValueError("image_paths must have the same length as y_true.")

    return n_true


#------------------------------------------------------------------------------
# Core image-level and pixel-level metrics
#------------------------------------------------------------------------------

# Compute ROC-AUC while handling edge cases cleanly.
def safe_roc_auc(y_true, y_score):
    y_true = as_int_1d(y_true)
    y_score = as_float_1d(y_score)

    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length.")

    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(roc_auc_score(y_true, y_score))


# Compute PR-AUC while handling edge cases cleanly.
def safe_pr_auc(y_true, y_score):
    y_true = as_int_1d(y_true)
    y_score = as_float_1d(y_score)

    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length.")

    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(average_precision_score(y_true, y_score))


# Flatten mask and heatmap lists so pixel ROC-AUC can be measured.
def pixel_roc_auc(masks_list, heatmaps_list):
    masks_list = as_list(masks_list)
    heatmaps_list = as_list(heatmaps_list)

    if len(masks_list) == 0 or len(heatmaps_list) == 0:
        return np.nan
    if len(masks_list) != len(heatmaps_list):
        raise ValueError("masks_list and heatmaps_list must have the same length.")

    flat_masks = []
    flat_heatmaps = []

    for mask_arr, heat_arr in zip(masks_list, heatmaps_list):
        mask_arr = np.asarray(mask_arr)
        heat_arr = np.asarray(heat_arr, dtype=np.float32)

        if mask_arr.shape != heat_arr.shape:
            raise ValueError("Every mask and heatmap must have the same shape.")

        flat_masks.append((mask_arr > 0).astype(np.int64).reshape(-1))
        flat_heatmaps.append(heat_arr.reshape(-1).astype(np.float32))

    y_true = np.concatenate(flat_masks).astype(np.int64)
    y_score = np.concatenate(flat_heatmaps).astype(np.float32)

    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(roc_auc_score(y_true, y_score))


# Convert anomaly scores into labels and compute threshold metrics.
def threshold_metrics(y_true, y_score, threshold):
    y_true = as_int_1d(y_true)
    y_score = as_float_1d(y_score)
    threshold = float(threshold)

    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length.")

    y_pred = (y_score >= threshold).astype(np.int64)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    n_total = int(len(y_true))
    n_anom = int(np.sum(y_true == 1))
    n_normal = int(np.sum(y_true == 0))

    acc = float((tp + tn) / max(n_total, 1))
    fpr = float(fp / max(n_normal, 1))
    tpr = float(tp / max(n_anom, 1))

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "fpr": float(fpr),
        "tpr": float(tpr),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_total": n_total,
        "n_anom": n_anom,
        "n_normal": n_normal,
    }


# Compute the core image-level metrics dictionary used throughout the notebooks.
def image_metrics_dict(y_true, y_score):
    y_true = as_int_1d(y_true)
    y_score = as_float_1d(y_score)

    return {
        "n_images": int(len(y_true)),
        "n_anom": int(np.sum(y_true == 1)),
        "n_normal": int(np.sum(y_true == 0)),
        "image_roc_auc": safe_roc_auc(y_true, y_score),
        "image_pr_auc": safe_pr_auc(y_true, y_score),
        "score_mean": float(np.mean(y_score)) if len(y_score) > 0 else np.nan,
        "score_std": float(np.std(y_score)) if len(y_score) > 0 else np.nan,
        "score_min": float(np.min(y_score)) if len(y_score) > 0 else np.nan,
        "score_max": float(np.max(y_score)) if len(y_score) > 0 else np.nan,
    }


#------------------------------------------------------------------------------
# Per-image evaluation table builders
#------------------------------------------------------------------------------

# Build one compact per-image evaluation table from model outputs.
def build_eval_table(
    y_true,
    y_score,
    categories,
    image_paths=None,
    masks_list=None,
    heatmaps_list=None,
    model_name=None,
    split_name="test",
    extra_cols=None,
):
    n_rows = validate_basic_lengths(
        y_true=y_true,
        y_score=y_score,
        categories=categories,
        image_paths=image_paths,
    )

    y_true = as_int_1d(y_true)
    y_score = as_float_1d(y_score)
    categories = [str(x) for x in categories]

    if image_paths is None:
        image_paths = [f"image_{i:05d}" for i in range(n_rows)]
    else:
        image_paths = [str(x) for x in image_paths]

    rows = {
        "model": [str(model_name) if model_name is not None else None] * n_rows,
        "split": [str(split_name)] * n_rows,
        "row_idx": list(range(n_rows)),
        "category": categories,
        "image_path": image_paths,
        "label": y_true.astype(np.int64),
        "score": y_score.astype(np.float32),
    }

    if masks_list is not None:
        if len(masks_list) != n_rows:
            raise ValueError("masks_list must match the number of rows.")
        rows["mask"] = list(masks_list)
        rows["mask_positive"] = [int(np.asarray(m).sum() > 0) for m in masks_list]

    if heatmaps_list is not None:
        if len(heatmaps_list) != n_rows:
            raise ValueError("heatmaps_list must match the number of rows.")
        rows["heatmap"] = list(heatmaps_list)

    if extra_cols is not None:
        for col_name, col_values in dict(extra_cols).items():
            if len(col_values) != n_rows:
                raise ValueError(f"Extra column '{col_name}' must match the number of rows.")
            rows[str(col_name)] = list(col_values)

    df = pd.DataFrame(rows)
    return df


# Add threshold-based predictions and flags to an evaluation table.
def add_threshold_columns(eval_df, threshold, threshold_name=None, score_col="score"):
    if score_col not in eval_df.columns:
        raise ValueError(f"'{score_col}' column is missing from eval_df.")

    out = eval_df.copy()
    threshold = float(threshold)
    out["threshold"] = threshold
    out["threshold_name"] = str(threshold_name) if threshold_name is not None else None
    out["pred"] = (out[score_col].to_numpy() >= threshold).astype(np.int64)
    out["is_tp"] = ((out["pred"] == 1) & (out["label"] == 1)).astype(np.int64)
    out["is_tn"] = ((out["pred"] == 0) & (out["label"] == 0)).astype(np.int64)
    out["is_fp"] = ((out["pred"] == 1) & (out["label"] == 0)).astype(np.int64)
    out["is_fn"] = ((out["pred"] == 0) & (out["label"] == 1)).astype(np.int64)
    return out


#------------------------------------------------------------------------------
# Summary table builders
#------------------------------------------------------------------------------

# Summarise one evaluation table at overall level.
def summarise_eval_table(
    eval_df,
    model_name=None,
    threshold=None,
    threshold_name=None,
    score_col="score",
    label_col="label",
    mask_col="mask",
    heatmap_col="heatmap",
):
    if score_col not in eval_df.columns or label_col not in eval_df.columns:
        raise ValueError("eval_df must include the score and label columns.")

    y_true = eval_df[label_col].to_numpy().astype(np.int64)
    y_score = eval_df[score_col].to_numpy().astype(np.float32)

    row = image_metrics_dict(y_true, y_score)
    row["model"] = str(model_name) if model_name is not None else (
        eval_df["model"].iloc[0] if "model" in eval_df.columns and len(eval_df) > 0 else None
    )
    row["category"] = "__overall__"

    if mask_col in eval_df.columns and heatmap_col in eval_df.columns:
        row["pixel_roc_auc"] = pixel_roc_auc(
            masks_list=eval_df[mask_col].tolist(),
            heatmaps_list=eval_df[heatmap_col].tolist(),
        )
    else:
        row["pixel_roc_auc"] = np.nan

    if threshold is not None:
        row.update(threshold_metrics(y_true, y_score, threshold))
        row["threshold_name"] = str(threshold_name) if threshold_name is not None else None
    else:
        row["threshold"] = np.nan
        row["threshold_name"] = str(threshold_name) if threshold_name is not None else None
        row["precision"] = np.nan
        row["recall"] = np.nan
        row["f1"] = np.nan
        row["accuracy"] = np.nan
        row["fpr"] = np.nan
        row["tpr"] = np.nan
        row["tp"] = np.nan
        row["tn"] = np.nan
        row["fp"] = np.nan
        row["fn"] = np.nan

    return pd.DataFrame([row])


# Summarise one evaluation table at category level.
def category_metrics_table(
    eval_df,
    model_name=None,
    threshold=None,
    threshold_name=None,
    score_col="score",
    label_col="label",
    category_col="category",
    mask_col="mask",
    heatmap_col="heatmap",
):
    needed_cols = {score_col, label_col, category_col}
    missing = [c for c in needed_cols if c not in eval_df.columns]
    if len(missing) > 0:
        raise ValueError(f"eval_df is missing required columns: {missing}")

    rows = []
    for category_name, g in eval_df.groupby(category_col, sort=True):
        y_true = g[label_col].to_numpy().astype(np.int64)
        y_score = g[score_col].to_numpy().astype(np.float32)

        row = image_metrics_dict(y_true, y_score)
        row["model"] = str(model_name) if model_name is not None else (
            g["model"].iloc[0] if "model" in g.columns and len(g) > 0 else None
        )
        row["category"] = str(category_name)

        if mask_col in g.columns and heatmap_col in g.columns:
            row["pixel_roc_auc"] = pixel_roc_auc(
                masks_list=g[mask_col].tolist(),
                heatmaps_list=g[heatmap_col].tolist(),
            )
        else:
            row["pixel_roc_auc"] = np.nan

        if threshold is not None:
            row.update(threshold_metrics(y_true, y_score, threshold))
            row["threshold_name"] = str(threshold_name) if threshold_name is not None else None
        else:
            row["threshold"] = np.nan
            row["threshold_name"] = str(threshold_name) if threshold_name is not None else None
            row["precision"] = np.nan
            row["recall"] = np.nan
            row["f1"] = np.nan
            row["accuracy"] = np.nan
            row["fpr"] = np.nan
            row["tpr"] = np.nan
            row["tp"] = np.nan
            row["tn"] = np.nan
            row["fp"] = np.nan
            row["fn"] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    sort_cols = [c for c in ["model", "category"] if c in out.columns]
    if len(sort_cols) > 0:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


# Collapse a category-level table to one mean summary row per group.
def mean_metrics_table(
    category_df,
    group_cols=None,
    metric_cols=None,
    category_col="category",
):
    if group_cols is None:
        group_cols = [c for c in ["model", "threshold_name"] if c in category_df.columns]
    group_cols = list(group_cols)

    default_metric_cols = [
        "image_roc_auc",
        "image_pr_auc",
        "pixel_roc_auc",
        "precision",
        "recall",
        "f1",
        "fpr",
        "accuracy",
        "tpr",
        "score_mean",
        "score_std",
    ]

    if metric_cols is None:
        metric_cols = [c for c in default_metric_cols if c in category_df.columns]
    else:
        metric_cols = [c for c in metric_cols if c in category_df.columns]

    if len(category_df) == 0:
        return pd.DataFrame(columns=group_cols + metric_cols + ["n_categories"])

    if len(group_cols) == 0:
        group_iter = [((), category_df.copy())]
    else:
        group_iter = category_df.groupby(group_cols, dropna=False, sort=True)

    rows = []
    for group_key, g in group_iter:
        if len(group_cols) == 0:
            row = {}
        elif len(group_cols) == 1:
            row = {group_cols[0]: group_key}
        else:
            row = dict(zip(group_cols, group_key))

        row["n_categories"] = int(len(g))
        if category_col in g.columns:
            row["categories"] = ", ".join(sorted(g[category_col].astype(str).tolist()))

        for metric_name in metric_cols:
            row[metric_name] = safe_nanmean(g[metric_name].to_numpy())

        rows.append(row)

    out = pd.DataFrame(rows)
    if len(group_cols) > 0 and len(out) > 0:
        out = out.sort_values(group_cols).reset_index(drop=True)
    return out


# Build overall, category, and mean tables in one call.
def summarise_run_tables(
    eval_df,
    model_name=None,
    threshold=None,
    threshold_name=None,
    score_col="score",
    label_col="label",
    category_col="category",
    mask_col="mask",
    heatmap_col="heatmap",
):
    overall_df = summarise_eval_table(
        eval_df=eval_df,
        model_name=model_name,
        threshold=threshold,
        threshold_name=threshold_name,
        score_col=score_col,
        label_col=label_col,
        mask_col=mask_col,
        heatmap_col=heatmap_col,
    )

    category_df = category_metrics_table(
        eval_df=eval_df,
        model_name=model_name,
        threshold=threshold,
        threshold_name=threshold_name,
        score_col=score_col,
        label_col=label_col,
        category_col=category_col,
        mask_col=mask_col,
        heatmap_col=heatmap_col,
    )

    mean_df = mean_metrics_table(category_df)

    return {
        "overall": overall_df,
        "category": category_df,
        "mean": mean_df,
    }


#------------------------------------------------------------------------------
# Ranking and convenience helpers
#------------------------------------------------------------------------------

# Rank one summary table by the selected metric.
def add_metric_rank(summary_df, metric_col="image_pr_auc", ascending=False, rank_col=None):
    if metric_col not in summary_df.columns:
        raise ValueError(f"'{metric_col}' column is missing from summary_df.")

    out = summary_df.copy()
    if rank_col is None:
        rank_col = f"rank_{metric_col}"

    out[rank_col] = out[metric_col].rank(method="dense", ascending=ascending)
    out = out.sort_values([rank_col, metric_col], ascending=[True, ascending]).reset_index(drop=True)
    return out


# Return the best row from one summary table based on the selected metric.
def best_summary_row(summary_df, metric_col="image_pr_auc", ascending=False):
    ranked = add_metric_rank(summary_df, metric_col=metric_col, ascending=ascending)
    if len(ranked) == 0:
        return {}
    return ranked.iloc[0].to_dict()


# Merge multiple category-level tables then rebuild a mean table.
def combine_category_tables(category_tables, group_cols=None):
    category_tables = [df for df in category_tables if df is not None and len(df) > 0]
    if len(category_tables) == 0:
        return {
            "category": pd.DataFrame(),
            "mean": pd.DataFrame(),
        }

    category_df = pd.concat(category_tables, axis=0, ignore_index=True)
    mean_df = mean_metrics_table(category_df, group_cols=group_cols)
    return {
        "category": category_df,
        "mean": mean_df,
    }


#------------------------------------------------------------------------------
# Save / load helpers
#------------------------------------------------------------------------------

# Save one dataframe to CSV.
def save_table_csv(df, file_path, index=False):
    file_path = ensure_parent_dir(file_path)
    df.to_csv(file_path, index=index)
    return str(file_path)


# Load one dataframe from CSV.
def load_table_csv(file_path):
    return pd.read_csv(Path(file_path))


# Save one dictionary as JSON.
def save_metrics_json(obj, file_path):
    file_path = ensure_parent_dir(file_path)

    def _convert(value):
        if isinstance(value, dict):
            return {str(k): _convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_convert(v) for v in value]
        return json_safe_scalar(value)

    with open(file_path, "w") as f:
        json.dump(_convert(obj), f, indent=2)

    return str(file_path)


# Load one metrics JSON file.
def load_metrics_json(file_path):
    with open(Path(file_path), "r") as f:
        return json.load(f)
