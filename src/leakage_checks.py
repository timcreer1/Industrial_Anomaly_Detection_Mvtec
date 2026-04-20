"""Utility functions for exact leakage checks on MVTec AD split manifests.

This module keeps together the leakage-specific helpers that are shared across
multiple notebooks. It focuses on:
- exact path overlap checks across train / validation / test partitions
- exact file duplicate checks using MD5 hashes
- compact category-level leakage summary tables
- save / load helpers for leakage reports used by later notebooks

Dataset reading stays in data_utils.py and split creation stays in split_utils.py
so the roles of the modules stay clear.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

try:
    from data_utils import load_splits_json
except ImportError:
    from .data_utils import load_splits_json


#------------------------------------------------------------------------------
# Hash helpers
#------------------------------------------------------------------------------

# Compute an MD5 hash for one file path.
def md5_file(path, chunk_size=8192):
    path = Path(path)
    hasher = hashlib.md5()

    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            hasher.update(block)

    return hasher.hexdigest()


# Build a path-to-MD5 mapping for a list of file paths.
def build_md5_map(paths, chunk_size=8192):
    paths = [str(Path(path)) for path in paths]
    return {path: md5_file(path, chunk_size=chunk_size) for path in paths}


# Count how many duplicate hash groups exist within one split.
def count_duplicate_groups(md5_values):
    md5_values = list(md5_values)
    if len(md5_values) == 0:
        return 0

    series = pd.Series(md5_values, dtype="object")
    return int((series.value_counts() > 1).sum())


# Count how many unique hashes overlap across two different splits.
def count_cross_duplicates(md5_values_a, md5_values_b):
    return int(len(set(md5_values_a).intersection(set(md5_values_b))))


#------------------------------------------------------------------------------
# Split path helpers
#------------------------------------------------------------------------------

# Return the train, validation, and test image paths for one category.
def get_category_split_paths(splits, category):
    if category not in splits:
        raise KeyError(f"Category '{category}' not found in split manifest.")

    info = splits[category]
    train_paths = [str(Path(path)) for path in info.get("train_good", [])]
    val_paths = [str(Path(path)) for path in info.get("val_good", [])]
    test_paths = [str(Path(row["img_path"])) for row in info.get("test", [])]

    return {
        "train_good": sorted(train_paths),
        "val_good": sorted(val_paths),
        "test": sorted(test_paths),
    }


# Return all split image paths flattened across categories.
def get_all_split_paths(splits):
    all_paths = {
        "train_good": [],
        "val_good": [],
        "test": [],
    }

    for category in sorted(splits.keys()):
        split_paths = get_category_split_paths(splits, category)
        for split_name, paths in split_paths.items():
            all_paths[split_name].extend(paths)

    for split_name in all_paths:
        all_paths[split_name] = sorted(all_paths[split_name])

    return all_paths


#------------------------------------------------------------------------------
# Category-level checks
#------------------------------------------------------------------------------

# Run exact path overlap and exact duplicate checks for one category.
def check_category_leakage(splits, category, chunk_size=8192):
    split_paths = get_category_split_paths(splits, category)

    train_paths = split_paths["train_good"]
    val_paths = split_paths["val_good"]
    test_paths = split_paths["test"]

    train_val_path_overlap = len(set(train_paths).intersection(set(val_paths)))
    train_test_path_overlap = len(set(train_paths).intersection(set(test_paths)))
    val_test_path_overlap = len(set(val_paths).intersection(set(test_paths)))

    train_md5 = list(build_md5_map(train_paths, chunk_size=chunk_size).values())
    val_md5 = list(build_md5_map(val_paths, chunk_size=chunk_size).values())
    test_md5 = list(build_md5_map(test_paths, chunk_size=chunk_size).values())

    row = {
        "category": category,
        "train_good_n": len(train_paths),
        "val_good_n": len(val_paths),
        "test_n": len(test_paths),
        "train_val_path_overlap_n": int(train_val_path_overlap),
        "train_test_path_overlap_n": int(train_test_path_overlap),
        "val_test_path_overlap_n": int(val_test_path_overlap),
        "train_duplicate_groups_n": int(count_duplicate_groups(train_md5)),
        "val_duplicate_groups_n": int(count_duplicate_groups(val_md5)),
        "test_duplicate_groups_n": int(count_duplicate_groups(test_md5)),
        "train_val_md5_overlap_n": int(count_cross_duplicates(train_md5, val_md5)),
        "train_test_md5_overlap_n": int(count_cross_duplicates(train_md5, test_md5)),
        "val_test_md5_overlap_n": int(count_cross_duplicates(val_md5, test_md5)),
    }
    return row


# Build a category-level leakage summary table for the full split manifest.
def build_leakage_summary_table(splits, categories=None, chunk_size=8192):
    categories = sorted(splits.keys()) if categories is None else list(categories)

    rows = []
    for category in categories:
        rows.append(check_category_leakage(splits, category, chunk_size=chunk_size))

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return pd.DataFrame(
            columns=[
                "category",
                "train_good_n",
                "val_good_n",
                "test_n",
                "train_val_path_overlap_n",
                "train_test_path_overlap_n",
                "val_test_path_overlap_n",
                "train_duplicate_groups_n",
                "val_duplicate_groups_n",
                "test_duplicate_groups_n",
                "train_val_md5_overlap_n",
                "train_test_md5_overlap_n",
                "val_test_md5_overlap_n",
            ]
        )

    return df.sort_values("category").reset_index(drop=True)


#------------------------------------------------------------------------------
# Aggregate checks and reports
#------------------------------------------------------------------------------

# Return True only if all numeric leakage checks are zero.
def leakage_table_all_zero(df_leakage):
    if len(df_leakage) == 0:
        return True

    numeric_cols = [col for col in df_leakage.columns if col != "category" and col.endswith("_n")]
    if len(numeric_cols) == 0:
        return True

    return bool((df_leakage[numeric_cols].fillna(0).to_numpy() == 0).all())


# Build a small totals dictionary from the leakage summary table.
def leakage_totals(df_leakage):
    if len(df_leakage) == 0:
        return {
            "category_n": 0,
            "train_good_total": 0,
            "val_good_total": 0,
            "test_total": 0,
            "path_overlap_total": 0,
            "duplicate_group_total": 0,
            "cross_split_md5_overlap_total": 0,
        }

    totals = {
        "category_n": int(df_leakage["category"].nunique()),
        "train_good_total": int(df_leakage["train_good_n"].sum()),
        "val_good_total": int(df_leakage["val_good_n"].sum()),
        "test_total": int(df_leakage["test_n"].sum()),
        "path_overlap_total": int(
            df_leakage[
                [
                    "train_val_path_overlap_n",
                    "train_test_path_overlap_n",
                    "val_test_path_overlap_n",
                ]
            ].sum().sum()
        ),
        "duplicate_group_total": int(
            df_leakage[
                [
                    "train_duplicate_groups_n",
                    "val_duplicate_groups_n",
                    "test_duplicate_groups_n",
                ]
            ].sum().sum()
        ),
        "cross_split_md5_overlap_total": int(
            df_leakage[
                [
                    "train_val_md5_overlap_n",
                    "train_test_md5_overlap_n",
                    "val_test_md5_overlap_n",
                ]
            ].sum().sum()
        ),
    }
    return totals


# Build a leakage report dictionary used by downstream notebooks.
def build_leakage_report(splits, categories=None, chunk_size=8192):
    df_leakage = build_leakage_summary_table(
        splits=splits,
        categories=categories,
        chunk_size=chunk_size,
    )

    report = {
        "checked_categories": sorted(df_leakage["category"].tolist()) if len(df_leakage) > 0 else [],
        "all_checks_zero": leakage_table_all_zero(df_leakage),
        "totals": leakage_totals(df_leakage),
        "rows": df_leakage.to_dict(orient="records"),
    }
    return report


# Raise an error if any leakage check is non-zero.
def assert_no_leakage(report):
    if not bool(report.get("all_checks_zero", False)):
        raise AssertionError(
            "Leakage checks reported non-zero overlaps or duplicate groups. "
            "Inspect the saved leakage summary before continuing."
        )
    return True


# Convert a report dictionary back into a leakage summary table.
def leakage_report_to_table(report):
    rows = report.get("rows", [])
    if len(rows) == 0:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("category").reset_index(drop=True)


#------------------------------------------------------------------------------
# Save and load helpers
#------------------------------------------------------------------------------

# Save both the leakage summary CSV and the JSON report.
def save_leakage_outputs(df_leakage, report, summary_csv_path=None, report_json_path=None):
    if summary_csv_path is not None:
        summary_csv_path = Path(summary_csv_path)
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_leakage.to_csv(summary_csv_path, index=False)

    if report_json_path is not None:
        report_json_path = Path(report_json_path)
        report_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


# Load a saved leakage report JSON file.
def load_leakage_report(report_json_path, validate=False):
    with open(Path(report_json_path), "r", encoding="utf-8") as f:
        report = json.load(f)

    if validate:
        validate_leakage_report(report)
    return report


# Check that a leakage report has the expected minimal structure.
def validate_leakage_report(report):
    required_keys = ["checked_categories", "all_checks_zero", "rows"]
    missing = [key for key in required_keys if key not in report]
    if len(missing) > 0:
        raise ValueError(f"Leakage report is missing required keys: {missing}")

    rows = report.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("Leakage report field 'rows' must be a list.")

    return True


# Load a split manifest JSON path, run leakage checks, and return both outputs.
def run_leakage_checks_from_split_json(split_json_path, categories=None, chunk_size=8192):
    splits = load_splits_json(split_json_path)
    df_leakage = build_leakage_summary_table(
        splits=splits,
        categories=categories,
        chunk_size=chunk_size,
    )
    report = build_leakage_report(
        splits=splits,
        categories=categories,
        chunk_size=chunk_size,
    )
    return df_leakage, report
