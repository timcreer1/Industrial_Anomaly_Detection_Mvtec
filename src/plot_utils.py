"""Plotting helpers for the MVTec industrial anomaly detection project.

This module keeps together lightweight matplotlib-based plotting utilities used
across the notebook pipeline. The helpers are designed for the project figures
that compare:
- baseline and main model performance
- category-level heatmaps
- accuracy versus efficiency trade-offs
- threshold policy comparisons and dense threshold sweeps
- ablation results
- generic training curves and compact summary tables

The goal is to keep figure generation consistent across the GitHub repo without
pulling in heavy styling dependencies. All functions save figures cleanly and
return the created figure/axes so notebook code can continue customising them
when needed.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#------------------------------------------------------------------------------
# Small path and dataframe helpers
#------------------------------------------------------------------------------

# Create the parent directory for one output file path.
def ensure_parent_dir(file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


# Return a defensive dataframe copy.
def copy_df(df):
    if df is None:
        return pd.DataFrame()
    return pd.DataFrame(df).copy()


# Keep only rows that contain the requested columns.
def require_columns(df, required_columns):
    df = copy_df(df)
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


# Convert one column to numeric, coercing invalid values to NaN.
def to_numeric_series(df, column):
    return pd.to_numeric(df[column], errors="coerce")


# Return a compact display name for a model if a mapping is provided.
def map_model_names(series, name_map=None):
    if name_map is None:
        return series.astype(str)
    return series.astype(str).map(lambda x: name_map.get(x, x))


# Sort a dataframe by an optional custom category order.
def sort_by_order(df, column, order=None):
    df = copy_df(df)
    if order is None:
        return df.sort_values(column).reset_index(drop=True)

    order_map = {value: idx for idx, value in enumerate(order)}
    df["_sort_key"] = df[column].map(lambda x: order_map.get(x, 10**9))
    df = df.sort_values(["_sort_key", column]).drop(columns=["_sort_key"])
    return df.reset_index(drop=True)


#------------------------------------------------------------------------------
# Generic figure helpers
#------------------------------------------------------------------------------

# Save one matplotlib figure and optionally close it afterwards.
def save_figure(fig, file_path, dpi=180, close=True, tight=True):
    file_path = ensure_parent_dir(file_path)
    if tight:
        fig.tight_layout()
    fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return str(file_path)


# Add value labels above bar containers.
def add_bar_labels(ax, fmt="{:.3f}", rotation=0, fontsize=9, y_pad_frac=0.01):
    y_min, y_max = ax.get_ylim()
    y_range = max(y_max - y_min, 1e-9)
    y_pad = y_range * y_pad_frac

    for patch in ax.patches:
        height = patch.get_height()
        if not np.isfinite(height):
            continue
        x = patch.get_x() + (patch.get_width() / 2.0)
        y = height + y_pad if height >= 0 else height - y_pad
        va = "bottom" if height >= 0 else "top"
        ax.text(
            x,
            y,
            fmt.format(height),
            ha="center",
            va=va,
            rotation=rotation,
            fontsize=fontsize,
        )


# Add text labels to scatter points using a dataframe row order.
def annotate_points(ax, df, x_col, y_col, label_col, fontsize=9, dx=0.0, dy=0.0):
    for _, row in copy_df(df).iterrows():
        x = row[x_col]
        y = row[y_col]
        label = str(row[label_col])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        ax.text(x + dx, y + dy, label, fontsize=fontsize)


# Build one compact dataframe preview image as a figure.
def plot_table_figure(
    df,
    title=None,
    max_rows=12,
    figsize=(10, 0.55),
    output_path=None,
):
    df = copy_df(df).head(max_rows)

    n_rows = max(len(df), 1)
    fig_h = max(1.2, figsize[1] * (n_rows + 2))
    fig, ax = plt.subplots(figsize=(figsize[0], fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)

    if title:
        ax.set_title(title)

    if output_path is not None:
        save_figure(fig, output_path)
    return fig, ax


#------------------------------------------------------------------------------
# Training-curve helpers
#------------------------------------------------------------------------------

# Plot one or more training curves from a dataframe or dict-like input.
def plot_training_curves(
    history_df,
    x_col="epoch",
    y_cols=None,
    title=None,
    xlabel=None,
    ylabel="Loss",
    figsize=(8, 4.5),
    output_path=None,
):
    history_df = copy_df(history_df)

    if history_df.empty:
        raise ValueError("history_df is empty.")

    if x_col not in history_df.columns:
        history_df[x_col] = np.arange(1, len(history_df) + 1)

    if y_cols is None:
        y_cols = [c for c in history_df.columns if c != x_col]

    fig, ax = plt.subplots(figsize=figsize)

    for col in y_cols:
        if col not in history_df.columns:
            continue
        y = pd.to_numeric(history_df[col], errors="coerce")
        ax.plot(history_df[x_col], y, marker="o", linewidth=1.8, label=col)

    ax.set_xlabel(xlabel or x_col.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if len(y_cols) > 1:
        ax.legend()
    ax.grid(alpha=0.25)

    if output_path is not None:
        save_figure(fig, output_path)
    return fig, ax


#------------------------------------------------------------------------------
# Performance-comparison plots
#------------------------------------------------------------------------------

# Plot a single-metric model comparison as a labelled bar chart.
def plot_metric_bar(
    df,
    model_col="model_name",
    metric_col="image_pr_auc",
    title=None,
    xlabel=None,
    ylabel=None,
    model_order=None,
    name_map=None,
    figsize=(9, 4.8),
    rotation=25,
    add_labels=True,
    output_path=None,
):
    df = require_columns(df, [model_col, metric_col])
    df = sort_by_order(df, model_col, model_order)
    df[model_col] = map_model_names(df[model_col], name_map)
    df[metric_col] = to_numeric_series(df, metric_col)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(df[model_col], df[metric_col])
    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or metric_col.replace("_", " ").upper())
    if title:
        ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha="right")

    if add_labels:
        add_bar_labels(ax)

    if output_path is not None:
        save_figure(fig, output_path)
    return fig, ax


# Plot a grouped bar chart for several metrics across models.
def plot_grouped_metric_bars(
    df,
    model_col="model_name",
    metric_cols=None,
    title=None,
    model_order=None,
    name_map=None,
    figsize=(10, 5),
    rotation=20,
    output_path=None,
):
    df = copy_df(df)
    if metric_cols is None:
        metric_cols = ["image_roc_auc", "image_pr_auc", "pixel_roc_auc"]

    require_columns(df, [model_col] + list(metric_cols))
    df = sort_by_order(df, model_col, model_order)
    df[model_col] = map_model_names(df[model_col], name_map)

    for col in metric_cols:
        df[col] = to_numeric_series(df, col)

    x = np.arange(len(df))
    width = 0.8 / max(len(metric_cols), 1)

    fig, ax = plt.subplots(figsize=figsize)

    for idx, metric_col in enumerate(metric_cols):
        ax.bar(x + (idx * width), df[metric_col], width=width, label=metric_col)

    ax.set_xticks(x + width * (len(metric_cols) - 1) / 2)
    ax.set_xticklabels(df[model_col], rotation=rotation, ha="right")
    ax.set_ylabel("Metric value")
    if title:
        ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend()

    if output_path is not None:
        save_figure(fig, output_path)
    return fig, ax


# Plot category by model heatmap using only matplotlib.
def plot_category_heatmap(
    df,
    row_col="category",
    col_col="model_name",
    value_col="image_pr_auc",
    title=None,
    row_order=None,
    col_order=None,
    name_map=None,
    figsize=(10, 6),
    annotate=True,
    value_fmt="{:.3f}",
    output_path=None,
):
    df = require_columns(df, [row_col, col_col, value_col])
    df = copy_df(df)
    df[col_col] = map_model_names(df[col_col], name_map)
    df[value_col] = to_numeric_series(df, value_col)

    pivot = df.pivot_table(index=row_col, columns=col_col, values=value_col, aggfunc="mean")
    if row_order is not None:
        pivot = pivot.reindex(row_order)
    if col_order is not None:
        mapped_col_order = [name_map.get(x, x) for x in col_order] if name_map else col_order
        pivot = pivot.reindex(columns=mapped_col_order)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(list(pivot.columns), rotation=30, ha="right")
    ax.set_yticklabels(list(pivot.index))

    if title:
        ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_col.replace("_", " ").upper())

    if annotate:
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.iloc[i, j]
                if pd.isna(value):
                    continue
                ax.text(j, i, value_fmt.format(value), ha="center", va="center", fontsize=8)

    if output_path is not None:
        save_figure(fig, output_path)
    return fig, ax, pivot


# Plot one scatter for accuracy versus efficiency.
def plot_accuracy_vs_efficiency(
    df,
    x_col="seconds_per_image",
    y_col="image_pr_auc",
    label_col="model_name",
    size_col=None,
    title=None,
    name_map=None,
    figsize=(8, 5),
    output_path=None,
):
    df = require_columns(df, [x_col, y_col, label_col])
    df = copy_df(df)
    df[x_col] = to_numeric_series(df, x_col)
    df[y_col] = to_numeric_series(df, y_col)
    df[label_col] = map_model_names(df[label_col], name_map)

    if size_col and size_col in df.columns:
        sizes = pd.to_numeric(df[size_col], errors="coerce").fillna(1.0).to_numpy()
        sizes = np.clip(sizes, a_min=1e-9, a_max=None)
        sizes = 150 * (sizes / sizes.max())
    else:
        sizes = np.full(len(df), 80.0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df[x_col], df[y_col], s=sizes)

    annotate_points(ax, df, x_col, y_col, label_col, fontsize=9, dx=0.001, dy=0.001)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").upper())
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.25)

    if output_path is not None:
        save_figure(fig, output_path)
    return fig, ax


#------------------------------------------------------------------------------
# Threshold plots
#------------------------------------------------------------------------------

# Plot grouped threshold metrics by policy for one or more models.
def plot_threshold_policy_comparison(
    df,
    policy_col="policy_name",
    model_col="model_name",
    metric_cols=None,
    title=None,
    policy_order=None,
    model_filter=None,
    name_map=None,
    figsize=(12, 5),
    output_path=None,
):
    df = copy_df(df)

    if metric_cols is None:
        metric_cols = ["precision", "recall", "f1", "fpr"]

    require_columns(df, [policy_col, model_col] + list(metric_cols))

    if model_filter is not None:
        model_filter = list(model_filter)
        df = df[df[model_col].isin(model_filter)].copy()

    if policy_order is not None:
        df = sort_by_order(df, policy_col, policy_order)

    df[model_col] = map_model_names(df[model_col], name_map)

    fig, axes = plt.subplots(1, len(metric_cols), figsize=figsize, squeeze=False)
    axes = axes.ravel()

    for ax, metric_col in zip(axes, metric_cols):
        pivot = df.pivot_table(index=policy_col, columns=model_col, values=metric_col, aggfunc="mean")
        pivot = pivot.copy()
        x = np.arange(len(pivot.index))
        width = 0.8 / max(len(pivot.columns), 1)

        for idx, col in enumerate(pivot.columns):
            ax.bar(x + (idx * width), pivot[col].to_numpy(), width=width, label=col)

        ax.set_xticks(x + width * (len(pivot.columns) - 1) / 2)
        ax.set_xticklabels(list(pivot.index), rotation=20, ha="right")
        ax.set_title(metric_col.upper())
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    if title:
        fig.suptitle(title)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))

    if output_path is not None:
        save_figure(fig, output_path)
    return fig, axes


# Plot dense threshold sweeps for one or more models.
def plot_threshold_sweep(
    df,
    threshold_col="threshold",
    y_cols=None,
    model_col="model_name",
    model_filter=None,
    title=None,
    name_map=None,
    figsize=(12, 4.5),
    output_path=None,
):
    df = copy_df(df)

    if y_cols is None:
        y_cols = ["precision", "recall", "f1"]

    require_columns(df, [threshold_col, model_col] + list(y_cols))

    if model_filter is not None:
        df = df[df[model_col].isin(list(model_filter))].copy()

    df[model_col] = map_model_names(df[model_col], name_map)
    df[threshold_col] = to_numeric_series(df, threshold_col)

    fig, axes = plt.subplots(1, len(y_cols), figsize=figsize, squeeze=False)
    axes = axes.ravel()

    for ax, metric_col in zip(axes, y_cols):
        for model_name, group_df in df.groupby(model_col):
            group_df = group_df.sort_values(threshold_col)
            ax.plot(
                group_df[threshold_col],
                pd.to_numeric(group_df[metric_col], errors="coerce"),
                linewidth=2.0,
                label=model_name,
            )

        ax.set_xlabel("Threshold")
        ax.set_ylabel(metric_col.upper())
        ax.set_title(metric_col.upper())
        ax.grid(alpha=0.25)

    if title:
        fig.suptitle(title)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))

    if output_path is not None:
        save_figure(fig, output_path)
    return fig, axes


#------------------------------------------------------------------------------
# Ablation plots
#------------------------------------------------------------------------------

# Plot one ablation comparison where the x-axis is one setting variable.
def plot_ablation_line(
    df,
    x_col,
    y_col="image_pr_auc",
    group_col=None,
    title=None,
    x_order=None,
    name_map=None,
    figsize=(8, 4.5),
    output_path=None,
):
    df = require_columns(df, [x_col, y_col] + ([group_col] if group_col else []))
    df = copy_df(df)
    df[y_col] = to_numeric_series(df, y_col)
    df = sort_by_order(df, x_col, x_order)

    fig, ax = plt.subplots(figsize=figsize)

    if group_col:
        df[group_col] = map_model_names(df[group_col], name_map)
        for group_name, group_df in df.groupby(group_col):
            group_df = sort_by_order(group_df, x_col, x_order)
            ax.plot(group_df[x_col].astype(str), group_df[y_col], marker="o", linewidth=2, label=group_name)
        ax.legend()
    else:
        ax.plot(df[x_col].astype(str), df[y_col], marker="o", linewidth=2)

    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel(y_col.replace("_", " ").upper())
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.25)

    if output_path is not None:
        save_figure(fig, output_path)
    return fig, ax


# Plot a 3-panel ablation summary for augmentation, layer, and coreset ratio.
def plot_ablation_summary(
    df,
    setting_col="ablation_name",
    value_col="setting_value",
    metric_col="image_pr_auc",
    title=None,
    figsize=(13, 4.5),
    output_path=None,
):
    df = require_columns(df, [setting_col, value_col, metric_col])
    df = copy_df(df)
    df[metric_col] = to_numeric_series(df, metric_col)

    settings = list(df[setting_col].dropna().astype(str).unique())
    n_panels = max(len(settings), 1)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize, squeeze=False)
    axes = axes.ravel()

    for ax, setting_name in zip(axes, settings):
        sub = df[df[setting_col].astype(str) == setting_name].copy()
        sub = sub.sort_values(value_col)
        ax.plot(sub[value_col].astype(str), sub[metric_col], marker="o", linewidth=2)
        ax.set_title(setting_name.replace("_", " ").title())
        ax.set_xlabel("Setting")
        ax.set_ylabel(metric_col.replace("_", " ").upper())
        ax.grid(alpha=0.25)

    # Hide any unused axes if present.
    for ax in axes[len(settings):]:
        ax.axis("off")

    if title:
        fig.suptitle(title)

    if output_path is not None:
        save_figure(fig, output_path)
    return fig, axes


#------------------------------------------------------------------------------
# Convenience wrappers for the most common project figures
#------------------------------------------------------------------------------

# Plot the main project comparison with the common three metrics.
def plot_main_results_figure(
    mean_df,
    model_col="model_name",
    title="Main Results",
    model_order=None,
    name_map=None,
    output_path=None,
):
    return plot_grouped_metric_bars(
        mean_df,
        model_col=model_col,
        metric_cols=["image_roc_auc", "image_pr_auc", "pixel_roc_auc"],
        title=title,
        model_order=model_order,
        name_map=name_map,
        figsize=(10.5, 5),
        output_path=output_path,
    )


# Plot the baseline comparison figure used in the report/readme.
def plot_baseline_results_figure(
    mean_df,
    model_col="model_name",
    title="Baseline Mean Results",
    model_order=None,
    name_map=None,
    output_path=None,
):
    return plot_grouped_metric_bars(
        mean_df,
        model_col=model_col,
        metric_cols=["image_roc_auc", "image_pr_auc", "pixel_roc_auc"],
        title=title,
        model_order=model_order,
        name_map=name_map,
        figsize=(9.8, 4.8),
        output_path=output_path,
    )


# Plot the main category heatmap used in the report/readme.
def plot_main_category_heatmap(
    category_df,
    title="Category Image PR-AUC Heatmap",
    row_order=None,
    col_order=None,
    name_map=None,
    output_path=None,
):
    return plot_category_heatmap(
        category_df,
        row_col="category",
        col_col="model_name",
        value_col="image_pr_auc",
        title=title,
        row_order=row_order,
        col_order=col_order,
        name_map=name_map,
        figsize=(10.5, 6),
        annotate=True,
        output_path=output_path,
    )
