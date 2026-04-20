"""Utility functions for building and saving MVTec AD split manifests.

This module keeps together the split-specific helpers that are shared across
multiple notebooks. It focuses on:
- deterministic train / validation splits for normal training images
- building a full split manifest for all active categories
- converting split manifests into flat tables for reporting
- lightweight save / load helpers for downstream notebooks

Dataset reading helpers live in data_utils.py and leakage-specific checks should
stay in leakage_checks.py so the roles of the modules stay clear.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from data_utils import (
        resolve_mvtec_dir,
        get_categories,
        get_train_good_paths,
        get_test_items,
        save_splits_json,
        load_splits_json,
        splits_to_summary_table,
    )
except ImportError:
    from .data_utils import (
        resolve_mvtec_dir,
        get_categories,
        get_train_good_paths,
        get_test_items,
        save_splits_json,
        load_splits_json,
        splits_to_summary_table,
    )


#------------------------------------------------------------------------------
# Seed and split helpers
#------------------------------------------------------------------------------

# Build a stable integer seed from the global seed and category name.
def stable_category_seed(seed, category):
    key = f"{seed}_{category}".encode("utf-8")
    return int(hashlib.md5(key).hexdigest()[:8], 16)


# Return a cleaned validation fraction that stays inside a safe range.
def clean_val_frac(val_frac):
    val_frac = float(val_frac)
    if val_frac <= 0 or val_frac >= 1:
        raise ValueError("val_frac must be strictly between 0 and 1.")
    return val_frac


# Split a list of normal training image paths into train and validation sets.
def split_train_val_paths(paths, val_frac=0.10, seed=42, category="unknown"):
    paths = [str(Path(path)) for path in paths]
    paths = sorted(paths)
    val_frac = clean_val_frac(val_frac)

    if len(paths) < 2:
        raise ValueError(
            f"Category '{category}' does not have enough normal training images to split."
        )

    rng = np.random.default_rng(stable_category_seed(seed, category))
    order = rng.permutation(len(paths))

    n_val = max(1, int(round(val_frac * len(paths))))
    n_val = min(n_val, len(paths) - 1)

    val_idx = sorted(order[:n_val].tolist())
    train_idx = sorted(order[n_val:].tolist())

    train_paths = [paths[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]
    return train_paths, val_paths


#------------------------------------------------------------------------------
# Manifest builders
#------------------------------------------------------------------------------

# Build the split manifest for one category.
def build_category_split(mvtec_dir, category, val_frac=0.10, seed=42):
    mvtec_dir = Path(mvtec_dir)
    category_dir = mvtec_dir / category

    train_good_all = get_train_good_paths(category_dir)
    train_good_paths, val_good_paths = split_train_val_paths(
        train_good_all,
        val_frac=val_frac,
        seed=seed,
        category=category,
    )

    test_items = get_test_items(category_dir, category)

    split_info = {
        "train_good": train_good_paths,
        "val_good": val_good_paths,
        "test": test_items,
    }
    return split_info


# Build the full split manifest across all requested categories.
def build_split_manifest(mvtec_dir=None, categories=None, val_frac=0.10, seed=42):
    mvtec_dir = resolve_mvtec_dir(mvtec_dir)
    categories = get_categories(mvtec_dir) if categories is None else list(categories)

    manifest = {}
    for category in categories:
        manifest[category] = build_category_split(
            mvtec_dir=mvtec_dir,
            category=category,
            val_frac=val_frac,
            seed=seed,
        )

    return manifest


#------------------------------------------------------------------------------
# Flat reporting tables
#------------------------------------------------------------------------------

# Convert the split manifest into one row per file for reporting and checks.
def split_manifest_to_rows(splits):
    rows = []

    for category, info in splits.items():
        for img_path in info.get("train_good", []):
            rows.append({
                "category": category,
                "split": "train_good",
                "img_path": str(img_path),
                "label": 0,
                "defect_type": "good",
                "mask_path": None,
            })

        for img_path in info.get("val_good", []):
            rows.append({
                "category": category,
                "split": "val_good",
                "img_path": str(img_path),
                "label": 0,
                "defect_type": "good",
                "mask_path": None,
            })

        for item in info.get("test", []):
            rows.append({
                "category": category,
                "split": "test",
                "img_path": str(item["img_path"]),
                "label": int(item["label"]),
                "defect_type": item.get("defect_type", "good"),
                "mask_path": item.get("mask_path"),
            })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return pd.DataFrame(
            columns=["category", "split", "img_path", "label", "defect_type", "mask_path"]
        )

    df = df.sort_values(["category", "split", "img_path"]).reset_index(drop=True)
    return df


# Convert the split manifest into one compact category-level summary table.
def split_manifest_to_summary(splits):
    return splits_to_summary_table(splits)


# Build a small totals dictionary that is easy to log in notebooks.
def split_manifest_totals(splits):
    df = split_manifest_to_summary(splits)
    if len(df) == 0:
        return {
            "category_n": 0,
            "train_good_total": 0,
            "val_good_total": 0,
            "test_total": 0,
            "test_good_total": 0,
            "test_anomaly_total": 0,
        }

    totals = {
        "category_n": int(df["category"].nunique()),
        "train_good_total": int(df["train_good_n"].sum()),
        "val_good_total": int(df["val_good_n"].sum()),
        "test_total": int(df["test_total_n"].sum()),
        "test_good_total": int(df["test_good_n"].sum()),
        "test_anomaly_total": int(df["test_anomaly_n"].sum()),
    }
    return totals


#------------------------------------------------------------------------------
# Split validation helpers
#------------------------------------------------------------------------------

# Check that each category has a non-empty train, validation, and test split.
def validate_split_manifest(splits):
    issues = []

    for category, info in splits.items():
        train_good = info.get("train_good", [])
        val_good = info.get("val_good", [])
        test_items = info.get("test", [])

        if len(train_good) == 0:
            issues.append(f"{category}: train_good is empty")
        if len(val_good) == 0:
            issues.append(f"{category}: val_good is empty")
        if len(test_items) == 0:
            issues.append(f"{category}: test is empty")

        if len(set(train_good).intersection(set(val_good))) > 0:
            issues.append(f"{category}: train_good and val_good overlap by path")

        for item in test_items:
            if "img_path" not in item:
                issues.append(f"{category}: test item missing img_path")
            if "label" not in item:
                issues.append(f"{category}: test item missing label")

    if len(issues) > 0:
        raise ValueError("Invalid split manifest:\n- " + "\n- ".join(issues))

    return True


# Return the category names in a stable order.
def list_split_categories(splits):
    return sorted(list(splits.keys()))


# Return just one requested split from one category.
def get_category_split_items(splits, category, split_name):
    if category not in splits:
        raise KeyError(f"Category '{category}' not found in split manifest.")
    if split_name not in ["train_good", "val_good", "test"]:
        raise ValueError("split_name must be one of: train_good, val_good, test")
    return splits[category][split_name]


#------------------------------------------------------------------------------
# Save and load helpers
#------------------------------------------------------------------------------

# Save the split manifest and optional flat reporting tables.
def save_split_outputs(splits, json_path, summary_csv_path=None, rows_csv_path=None):
    validate_split_manifest(splits)
    save_splits_json(splits, json_path)

    if summary_csv_path is not None:
        summary_csv_path = Path(summary_csv_path)
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
        split_manifest_to_summary(splits).to_csv(summary_csv_path, index=False)

    if rows_csv_path is not None:
        rows_csv_path = Path(rows_csv_path)
        rows_csv_path.parent.mkdir(parents=True, exist_ok=True)
        split_manifest_to_rows(splits).to_csv(rows_csv_path, index=False)


# Load the split manifest and optionally validate it.
def load_split_manifest(json_path, validate=True):
    splits = load_splits_json(json_path)
    if validate:
        validate_split_manifest(splits)
    return splits


#------------------------------------------------------------------------------
# Small convenience helper for notebook logging
#------------------------------------------------------------------------------

# Build a compact dictionary describing the split configuration.
def describe_split_config(categories, val_frac, seed):
    categories = list(categories)
    return {
        "category_n": len(categories),
        "categories": categories,
        "val_frac": float(val_frac),
        "seed": int(seed),
    }
