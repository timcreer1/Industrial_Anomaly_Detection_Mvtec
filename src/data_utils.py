"""Utility functions for reading and organising the MVTec AD dataset.

This module keeps together the dataset-specific helpers that are shared across
multiple notebooks. It focuses on:
- resolving the dataset root
- listing categories and image files
- matching anomaly images to ground-truth masks
- building consistent item dictionaries for train / val / test workflows
- lightweight dataset objects for PyTorch loaders

The split creation logic is intentionally left for split_utils.py so the roles of
both files stay clear.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


#------------------------------------------------------------------------------
# Dataset path helpers
#------------------------------------------------------------------------------

# Resolve the MVTec root from an explicit path, environment variable, or common
# Kaggle / local defaults.
def resolve_mvtec_dir(mvtec_dir=None, env_var="MVTEC_DIR"):
    candidates = []

    if mvtec_dir is not None:
        candidates.append(Path(mvtec_dir))

    env_value = os.environ.get(env_var)
    if env_value:
        candidates.append(Path(env_value))

    candidates.extend([
        Path("/kaggle/input/datasets/ipythonx/mvtec-ad"),
        Path("/kaggle/input/mvtec-ad"),
        Path("/kaggle/input/mvtec"),
        Path("./mvtec-ad"),
        Path("./data/mvtec-ad"),
        Path("../data/mvtec-ad"),
    ])

    for path in candidates:
        if path.exists() and path.is_dir():
            return path.resolve()

    checked = [str(p) for p in candidates]
    raise FileNotFoundError(
        "Could not resolve the MVTec AD dataset directory. "
        f"Checked: {checked}. Pass mvtec_dir explicitly or set {env_var}."
    )


# Return the category folders found under the dataset root.
def get_categories(mvtec_dir, sort_output=True):
    mvtec_dir = Path(mvtec_dir)
    categories = [p.name for p in mvtec_dir.iterdir() if p.is_dir()]
    return sorted(categories) if sort_output else categories


# Check that the main expected MVTec subfolders exist for one category.
def validate_category_structure(cat_dir):
    cat_dir = Path(cat_dir)
    expected = [cat_dir / "train", cat_dir / "test", cat_dir / "ground_truth"]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing expected MVTec folders for category '{cat_dir.name}': {missing}"
        )
    return True


#------------------------------------------------------------------------------
# File listing and mask helpers
#------------------------------------------------------------------------------

# List PNG images in one folder.
def list_pngs(folder_path):
    folder_path = Path(folder_path)
    if not folder_path.exists():
        return []
    return sorted(folder_path.glob("*.png"))


# List immediate subfolders in one folder.
def list_subdirs(folder_path):
    folder_path = Path(folder_path)
    if not folder_path.exists():
        return []
    return sorted([p for p in folder_path.iterdir() if p.is_dir()])


# Standardise the mask stem so image files and mask files match cleanly.
def mask_stem(mask_path):
    stem = Path(mask_path).stem
    return stem[:-5] if stem.endswith("_mask") else stem


# Build a lookup from image stem to the matching ground-truth mask path.
def build_mask_lookup(mask_paths):
    return {mask_stem(p): Path(p) for p in mask_paths}


# Inspect how masks are named inside one defect folder.
def infer_mask_style(mask_paths):
    mask_paths = list(mask_paths)
    if len(mask_paths) == 0:
        return "none"
    n_mask_suffix = sum(Path(p).stem.endswith("_mask") for p in mask_paths)
    return "stem_mask" if n_mask_suffix > 0 else "same_name"


# Match one anomaly image to its mask path.
def match_mask_for_image(img_path, mask_lookup):
    img_path = Path(img_path)
    return mask_lookup.get(img_path.stem)


# Pick a defect folder with usable image-mask pairs for qualitative plots.
def choose_example_defect(cat_dir):
    cat_dir = Path(cat_dir)
    candidates = []

    for defect_dir in list_subdirs(cat_dir / "test"):
        defect_name = defect_dir.name
        if defect_name == "good":
            continue

        img_paths = list_pngs(cat_dir / "test" / defect_name)
        mask_paths = list_pngs(cat_dir / "ground_truth" / defect_name)
        mask_lookup = build_mask_lookup(mask_paths)
        paired_n = sum(p.stem in mask_lookup for p in img_paths)

        if paired_n > 0:
            candidates.append((paired_n, defect_name, img_paths, mask_lookup))

    if len(candidates) == 0:
        return None, [], {}

    candidates = sorted(candidates, key=lambda x: (-x[0], x[1]))
    _, defect_name, img_paths, mask_lookup = candidates[0]
    return defect_name, img_paths, mask_lookup


#------------------------------------------------------------------------------
# Image and mask loading helpers
#------------------------------------------------------------------------------

# Load one RGB image as a PIL image.
def load_rgb_pil(path):
    return Image.open(path).convert("RGB")


# Load one RGB image as a NumPy array for plotting.
def load_rgb_np(path):
    return np.array(load_rgb_pil(path))


# Load one mask as a grayscale PIL image.
def load_mask_pil(path):
    return Image.open(path).convert("L")


# Load one mask as a NumPy array for plotting.
def load_mask_np(path):
    return np.array(load_mask_pil(path))


# Convert a mask path to a resized binary tensor used in evaluation.
def load_mask_tensor(mask_path, img_size):
    if mask_path is None:
        return torch.zeros((1, img_size, img_size), dtype=torch.float32)

    mask_img = load_mask_pil(mask_path).resize((img_size, img_size), resample=Image.NEAREST)
    mask_np = (np.array(mask_img, dtype=np.float32) > 0).astype(np.float32)
    return torch.from_numpy(mask_np)[None, :, :]


#------------------------------------------------------------------------------
# Item building helpers
#------------------------------------------------------------------------------

# Build the list of normal training image paths for one category.
def get_train_good_paths(cat_dir):
    cat_dir = Path(cat_dir)
    return list_pngs(cat_dir / "train" / "good")


# Build the list of good test image item dictionaries for one category.
def get_test_good_items(cat_dir, category):
    cat_dir = Path(cat_dir)
    items = []

    for img_path in list_pngs(cat_dir / "test" / "good"):
        items.append({
            "category": category,
            "defect_type": "good",
            "img_path": str(img_path),
            "label": 0,
            "mask_path": None,
        })

    return items


# Build the list of anomaly test item dictionaries for one category.
def get_test_anomaly_items(cat_dir, category):
    cat_dir = Path(cat_dir)
    items = []

    for defect_dir in list_subdirs(cat_dir / "test"):
        defect_name = defect_dir.name
        if defect_name == "good":
            continue

        img_paths = list_pngs(cat_dir / "test" / defect_name)
        mask_paths = list_pngs(cat_dir / "ground_truth" / defect_name)
        mask_lookup = build_mask_lookup(mask_paths)

        for img_path in img_paths:
            mask_path = match_mask_for_image(img_path, mask_lookup)
            items.append({
                "category": category,
                "defect_type": defect_name,
                "img_path": str(img_path),
                "label": 1,
                "mask_path": None if mask_path is None else str(mask_path),
            })

    return items


# Build the combined test set item dictionaries for one category.
def get_test_items(cat_dir, category):
    good_items = get_test_good_items(cat_dir, category)
    anomaly_items = get_test_anomaly_items(cat_dir, category)
    items = good_items + anomaly_items
    items = sorted(items, key=lambda x: (x["label"], x["defect_type"], x["img_path"]))
    return items


# Build a simple one-category summary used in audit tables.
def build_category_summary(cat_dir, category):
    cat_dir = Path(cat_dir)
    validate_category_structure(cat_dir)

    train_good_paths = get_train_good_paths(cat_dir)
    test_good_items = get_test_good_items(cat_dir, category)
    test_anomaly_items = get_test_anomaly_items(cat_dir, category)

    test_defect_folders = [d.name for d in list_subdirs(cat_dir / "test") if d.name != "good"]
    gt_defect_folders = [d.name for d in list_subdirs(cat_dir / "ground_truth")]
    mask_total_n = sum(len(list_pngs(cat_dir / "ground_truth" / defect)) for defect in gt_defect_folders)

    summary = {
        "category": category,
        "train_good_n": len(train_good_paths),
        "test_good_n": len(test_good_items),
        "test_anomaly_n": len(test_anomaly_items),
        "test_total_n": len(test_good_items) + len(test_anomaly_items),
        "test_defect_types_n": len(test_defect_folders),
        "test_defect_types": sorted(test_defect_folders),
        "ground_truth_defect_types_n": len(gt_defect_folders),
        "ground_truth_defect_types": sorted(gt_defect_folders),
        "masks_total_n": mask_total_n,
    }
    return summary


# Build a full dataset audit table across all categories.
def build_dataset_summary_table(mvtec_dir, categories=None):
    mvtec_dir = Path(mvtec_dir)
    categories = get_categories(mvtec_dir) if categories is None else list(categories)

    rows = []
    for category in categories:
        cat_dir = mvtec_dir / category
        rows.append(build_category_summary(cat_dir, category))

    df = pd.DataFrame(rows).sort_values("category").reset_index(drop=True)
    return df


#------------------------------------------------------------------------------
# Split manifest helpers
#------------------------------------------------------------------------------

# Save a split dictionary to JSON for later notebooks.
def save_splits_json(splits, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)


# Load a previously saved split manifest.
def load_splits_json(path):
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


# Convert the split dictionary into a compact reporting table.
def splits_to_summary_table(splits):
    rows = []
    for category, info in splits.items():
        test_items = info.get("test", [])
        rows.append({
            "category": category,
            "train_good_n": len(info.get("train_good", [])),
            "val_good_n": len(info.get("val_good", [])),
            "test_total_n": len(test_items),
            "test_good_n": sum(int(x["label"]) == 0 for x in test_items),
            "test_anomaly_n": sum(int(x["label"]) == 1 for x in test_items),
        })

    df = pd.DataFrame(rows).sort_values("category").reset_index(drop=True)
    return df


#------------------------------------------------------------------------------
# PyTorch dataset and loader helpers
#------------------------------------------------------------------------------

# Return images, labels, masks, and paths in a consistent dataset format.
class MvtecDataset(Dataset):
    def __init__(self, items, mode, img_tfm, img_size):
        self.items = list(items)
        self.mode = mode
        self.img_tfm = img_tfm
        self.img_size = img_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        # For train_good and val_good, the item is stored as just an image path.
        if self.mode in ["train_good", "val_good"]:
            img_path = item
            label = 0
            mask_path = None
        else:
            img_path = item["img_path"]
            label = int(item["label"])
            mask_path = item.get("mask_path")

        image = self.img_tfm(load_rgb_pil(img_path))
        mask = load_mask_tensor(mask_path, self.img_size)
        return image, int(label), mask, str(img_path)


# Create a DataLoader with settings matched to the current runtime.
def make_loader(items, mode, img_tfm, img_size, batch_size, shuffle, num_workers=0,
                pin_memory=False, persistent_workers=False, prefetch_factor=2):
    dataset = MvtecDataset(items=items, mode=mode, img_tfm=img_tfm, img_size=img_size)

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(dataset, **loader_kwargs)


# Build the train, validation, and test loaders for one category.
def make_split_loaders(splits, category, img_tfm, img_size, batch_size_train,
                       batch_size_test, input_kind=None, num_workers=0,
                       pin_memory=False, persistent_workers=False, prefetch_factor=2):
    del input_kind

    train_loader = make_loader(
        items=splits[category]["train_good"],
        mode="train_good",
        img_tfm=img_tfm,
        img_size=img_size,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    val_loader = make_loader(
        items=splits[category]["val_good"],
        mode="val_good",
        img_tfm=img_tfm,
        img_size=img_size,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    test_loader = make_loader(
        items=splits[category]["test"],
        mode="test",
        img_tfm=img_tfm,
        img_size=img_size,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    return train_loader, val_loader, test_loader


#------------------------------------------------------------------------------
# Small audit helpers for qualitative selection
#------------------------------------------------------------------------------

# Grab a small random sample of items for quick visual checks.
def pick_n(items, n, seed=None):
    items = list(items)
    if len(items) == 0:
        return []

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(items), size=min(n, len(items)), replace=False)
    return [items[i] for i in idx]
