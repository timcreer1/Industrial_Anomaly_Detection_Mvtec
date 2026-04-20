"""Heatmap and qualitative-visualisation helpers for the MVTec AD project.

This module keeps together the project-level utilities used to:
- normalise anomaly heatmaps for display
- convert tensors into display-ready RGB images
- resize heatmaps and masks to a target image size
- draw overlays, masks, and raw heatmaps on matplotlib axes
- build compact qualitative grids for reports and README figures
- save detailed prediction tables for failure-analysis style review

The goal is to keep the qualitative plotting logic consistent across the
baseline, SSL, threshold, and failure-analysis notebooks while staying simple
and notebook-friendly.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

try:
    from .transform_utils import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        tensor_to_numpy_image,
        mask_tensor_to_numpy,
    )
except Exception:
    from transform_utils import (  # type: ignore
        IMAGENET_MEAN,
        IMAGENET_STD,
        tensor_to_numpy_image,
        mask_tensor_to_numpy,
    )


#------------------------------------------------------------------------------
# Small path and array helpers
#------------------------------------------------------------------------------

# Create the parent directory for one output path.
def ensure_parent_dir(file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


# Convert one object to a 2D NumPy array.
def as_numpy_hw(array_like, index=0):
    if torch.is_tensor(array_like):
        arr = array_like.detach().cpu().numpy()
    else:
        arr = np.asarray(array_like)

    if arr.ndim == 4:
        arr = arr[index]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim != 2:
        raise ValueError("Expected a 2D array, 1HW tensor, HW array, or a batched equivalent.")

    return arr


# Return a clean (height, width) tuple.
def normalise_out_size(out_size):
    if isinstance(out_size, int):
        return int(out_size), int(out_size)
    if isinstance(out_size, (tuple, list)) and len(out_size) == 2:
        return int(out_size[0]), int(out_size[1])
    raise ValueError("out_size must be an int or a (height, width) pair.")


#------------------------------------------------------------------------------
# Core heatmap helpers
#------------------------------------------------------------------------------

# Normalise one array to the 0-1 range for display.
def norm_01(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x.astype(np.float32)

    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))
    if not np.isfinite(x_min) or not np.isfinite(x_max):
        return np.zeros_like(x, dtype=np.float32)
    if abs(x_max - x_min) < float(eps):
        return np.zeros_like(x, dtype=np.float32)

    out = (x - x_min) / (x_max - x_min + float(eps))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# Robust 0-1 normalisation using quantiles to reduce outlier dominance.
def norm_01_quantile(x, q_low=0.01, q_high=0.99, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x.astype(np.float32)

    lo = float(np.nanquantile(x, q_low))
    hi = float(np.nanquantile(x, q_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < float(eps):
        return np.zeros_like(x, dtype=np.float32)

    out = (x - lo) / (hi - lo + float(eps))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# Resize one heatmap to a target image size.
def resize_heatmap(heat_2d, out_size, resample=Image.BILINEAR):
    heat_2d = as_numpy_hw(heat_2d)
    out_h, out_w = normalise_out_size(out_size)

    pil_img = Image.fromarray(heat_2d.astype(np.float32), mode="F")
    pil_img = pil_img.resize((out_w, out_h), resample=resample)
    return np.asarray(pil_img, dtype=np.float32)


# Resize one binary mask to a target image size without introducing soft edges.
def resize_mask(mask_2d, out_size, threshold=0.5):
    mask_2d = as_numpy_hw(mask_2d).astype(np.float32)
    out_h, out_w = normalise_out_size(out_size)

    pil_img = Image.fromarray(mask_2d, mode="F")
    pil_img = pil_img.resize((out_w, out_h), resample=Image.NEAREST)
    arr = np.asarray(pil_img, dtype=np.float32)
    return (arr >= float(threshold)).astype(np.float32)


# Convert one anomaly heatmap to an image-sized NumPy array when needed.
def prepare_heatmap_for_display(heat_2d, image_shape=None, normalize=True, robust=False):
    heat_2d = as_numpy_hw(heat_2d).astype(np.float32)

    if image_shape is not None:
        if len(image_shape) == 3:
            target_hw = int(image_shape[0]), int(image_shape[1])
        elif len(image_shape) == 2:
            target_hw = int(image_shape[0]), int(image_shape[1])
        else:
            raise ValueError("image_shape must be a HW or HWC shape tuple.")

        if heat_2d.shape != target_hw:
            heat_2d = resize_heatmap(heat_2d, target_hw)

    if normalize:
        heat_2d = norm_01_quantile(heat_2d) if robust else norm_01(heat_2d)

    return heat_2d.astype(np.float32)


#------------------------------------------------------------------------------
# Display conversion helpers
#------------------------------------------------------------------------------

# Convert a project image tensor to a display-ready NumPy RGB array.
def tensor_to_display(img_tensor, mean=None, std=None):
    if not torch.is_tensor(img_tensor):
        raise TypeError("tensor_to_display expects a torch.Tensor input.")

    img = img_tensor.detach().cpu()
    if img.ndim == 4:
        img = img[0]
    if img.ndim != 3:
        raise ValueError("tensor_to_display expects a CHW or BCHW tensor.")

    arr_raw = img.permute(1, 2, 0).numpy().astype(np.float32)
    if arr_raw.min() < 0.0 or arr_raw.max() > 1.5:
        return tensor_to_numpy_image(
            img,
            mean=IMAGENET_MEAN if mean is None else mean,
            std=IMAGENET_STD if std is None else std,
            denormalize=True,
        )

    return np.clip(arr_raw, 0.0, 1.0)


# Convert a mask tensor or array to a clean binary display array.
def mask_to_display(mask_like, index=0, threshold=0.5):
    if torch.is_tensor(mask_like):
        mask = mask_tensor_to_numpy(mask_like, index=index)
    else:
        mask = as_numpy_hw(mask_like, index=index)
    return (np.asarray(mask, dtype=np.float32) >= float(threshold)).astype(np.float32)


#------------------------------------------------------------------------------
# Matplotlib axis helpers
#------------------------------------------------------------------------------

# Draw an image with an optional heatmap overlay.
def overlay(ax, img_tensor, heat_2d=None, title="", alpha=0.45, cmap="jet", robust=False):
    image_np = tensor_to_display(img_tensor)
    ax.imshow(image_np)

    if heat_2d is not None:
        heat_2d = prepare_heatmap_for_display(
            heat_2d,
            image_shape=image_np.shape,
            normalize=True,
            robust=robust,
        )
        ax.imshow(heat_2d, cmap=cmap, alpha=float(alpha))

    ax.set_title(title, fontsize=10)
    ax.axis("off")


# Draw a raw NumPy image with an optional heatmap overlay.
def overlay_np(ax, image_np, heat_2d=None, title="", alpha=0.45, cmap="jet", robust=False):
    image_np = np.asarray(image_np, dtype=np.float32)
    image_np = np.clip(image_np, 0.0, 1.0)
    ax.imshow(image_np)

    if heat_2d is not None:
        heat_2d = prepare_heatmap_for_display(
            heat_2d,
            image_shape=image_np.shape,
            normalize=True,
            robust=robust,
        )
        ax.imshow(heat_2d, cmap=cmap, alpha=float(alpha))

    ax.set_title(title, fontsize=10)
    ax.axis("off")


# Show a heatmap on its own axis.
def show_heatmap(ax, heat_2d, title="", cmap="jet", robust=False, add_colorbar=False):
    heat_2d = prepare_heatmap_for_display(heat_2d, normalize=True, robust=robust)
    im = ax.imshow(heat_2d, cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

    if add_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# Show a binary mask with a clean grayscale colourmap.
def show_mask(ax, mask_2d, title=""):
    mask_2d = mask_to_display(mask_2d)
    ax.imshow(mask_2d, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


# Show a plain RGB image without any overlay.
def show_image(ax, img_tensor, title=""):
    ax.imshow(tensor_to_display(img_tensor))
    ax.set_title(title, fontsize=10)
    ax.axis("off")


#------------------------------------------------------------------------------
# Overlay image builders
#------------------------------------------------------------------------------

# Return one standalone colour heatmap image.
def heatmap_to_rgb(heat_2d, cmap="jet", robust=False):
    heat_2d = prepare_heatmap_for_display(heat_2d, normalize=True, robust=robust)
    cmap_obj = plt.get_cmap(cmap)
    rgb = cmap_obj(heat_2d)[..., :3]
    return np.asarray(rgb, dtype=np.float32)


# Blend one RGB image with one anomaly heatmap and return a NumPy RGB image.
def overlay_image_heatmap(image_np, heat_2d, alpha=0.45, cmap="jet", robust=False):
    image_np = np.asarray(image_np, dtype=np.float32)
    image_np = np.clip(image_np, 0.0, 1.0)

    heat_rgb = heatmap_to_rgb(
        heat_2d,
        cmap=cmap,
        robust=robust,
    )

    if heat_rgb.shape[:2] != image_np.shape[:2]:
        heat_rgb = heatmap_to_rgb(
            resize_heatmap(heat_2d, image_np.shape[:2]),
            cmap=cmap,
            robust=robust,
        )

    alpha = float(alpha)
    out = (1.0 - alpha) * image_np + alpha * heat_rgb
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# Build one side-by-side qualitative panel as NumPy RGB arrays.
def build_triptych_arrays(img_tensor, heat_2d=None, mask_2d=None, alpha=0.45, cmap="jet", robust=False):
    image_np = tensor_to_display(img_tensor)
    overlay_np_img = None
    mask_np = None

    if heat_2d is not None:
        overlay_np_img = overlay_image_heatmap(
            image_np,
            heat_2d,
            alpha=alpha,
            cmap=cmap,
            robust=robust,
        )

    if mask_2d is not None:
        mask_np = mask_to_display(mask_2d)

    return {
        "image": image_np,
        "overlay": overlay_np_img,
        "mask": mask_np,
    }


#------------------------------------------------------------------------------
# Detailed prediction / failure-analysis helpers
#------------------------------------------------------------------------------

# Collect per-image qualitative outputs from one loader and score function.
def collect_detailed_preds(loader, score_fn):
    rows = []
    for images, labels, masks, paths in loader:
        scores, heats = score_fn(images)
        for idx in range(len(paths)):
            rows.append({
                "path": str(paths[idx]),
                "label": int(labels[idx].item()),
                "score": float(scores[idx]),
                "img_tensor": images[idx].detach().cpu(),
                "mask": masks[idx].squeeze(0).detach().cpu().numpy(),
                "heat": np.asarray(heats[idx], dtype=np.float32),
            })
    return rows


# Convert a list of detailed prediction rows into a compact DataFrame.
def detailed_preds_to_table(rows, threshold=None):
    out_rows = []
    threshold_val = None if threshold is None else float(threshold)

    for row in rows:
        score = float(row["score"])
        label = int(row["label"])
        pred = None if threshold_val is None else int(score >= threshold_val)

        bucket = None
        if pred is not None:
            if label == 1 and pred == 1:
                bucket = "TP"
            elif label == 0 and pred == 0:
                bucket = "TN"
            elif label == 0 and pred == 1:
                bucket = "FP"
            elif label == 1 and pred == 0:
                bucket = "FN"

        out_rows.append({
            "path": str(row["path"]),
            "label": label,
            "score": score,
            "threshold": threshold_val,
            "pred": pred,
            "bucket": bucket,
        })

    return pd.DataFrame(out_rows)


# Select the hardest examples from one prediction bucket.
def select_examples_by_bucket(table_df, bucket="FN", top_n=12, score_col="score"):
    if len(table_df) == 0:
        return table_df.copy()

    table_df = table_df.copy()
    bucket = str(bucket).upper()
    subset = table_df.loc[table_df["bucket"].astype(str).str.upper() == bucket].copy()

    if bucket in ["FN", "TP"]:
        subset = subset.sort_values(score_col, ascending=False)
    else:
        subset = subset.sort_values(score_col, ascending=False)

    return subset.head(int(top_n)).reset_index(drop=True)


#------------------------------------------------------------------------------
# Figure builders
#------------------------------------------------------------------------------

# Save a small qualitative grid from rows returned by collect_detailed_preds.
def save_qualitative_grid(
    rows,
    fig_path,
    n_examples=6,
    title=None,
    alpha=0.45,
    cmap="jet",
    robust=False,
    score_fmt="{:.4f}",
):
    rows = list(rows)[: int(n_examples)]
    if len(rows) == 0:
        raise ValueError("save_qualitative_grid received no rows to plot.")

    fig_path = ensure_parent_dir(fig_path)
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(11, 3.6 * n_rows))

    if n_rows == 1:
        axes = np.asarray([axes])

    for row_idx, row in enumerate(rows):
        img_title = Path(row.get("path", f"image_{row_idx}")).name
        score_text = score_fmt.format(float(row.get("score", np.nan)))
        label_text = int(row.get("label", 0))

        show_image(axes[row_idx, 0], row["img_tensor"], title=f"Image\n{img_title}")
        overlay(
            axes[row_idx, 1],
            row["img_tensor"],
            row.get("heat"),
            title=f"Overlay\nscore={score_text}, label={label_text}",
            alpha=alpha,
            cmap=cmap,
            robust=robust,
        )
        show_mask(axes[row_idx, 2], row.get("mask"), title="Mask")

    if title is not None:
        fig.suptitle(str(title), fontsize=13, y=0.995)

    fig.tight_layout(rect=[0, 0, 1, 0.985] if title is not None else None)
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return fig_path


# Save failure examples from a detailed row list and a selected table subset.
def save_failure_examples_figure(
    detailed_rows,
    selected_table,
    fig_path,
    title=None,
    alpha=0.45,
    cmap="jet",
    robust=False,
):
    row_lookup = {str(row["path"]): row for row in detailed_rows}
    selected_rows = []

    for _, table_row in selected_table.iterrows():
        path = str(table_row["path"])
        if path in row_lookup:
            selected_rows.append(row_lookup[path])

    if len(selected_rows) == 0:
        raise ValueError("No selected failure examples were found in the detailed row lookup.")

    return save_qualitative_grid(
        selected_rows,
        fig_path=fig_path,
        n_examples=len(selected_rows),
        title=title,
        alpha=alpha,
        cmap=cmap,
        robust=robust,
    )


# Save a simple single-overlay figure for README or report use.
def save_single_overlay_figure(
    img_tensor,
    heat_2d,
    mask_2d,
    fig_path,
    title=None,
    alpha=0.45,
    cmap="jet",
    robust=False,
):
    fig_path = ensure_parent_dir(fig_path)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    show_image(axes[0], img_tensor, title="Image")
    overlay(axes[1], img_tensor, heat_2d, title="Overlay", alpha=alpha, cmap=cmap, robust=robust)
    show_mask(axes[2], mask_2d, title="Mask")

    if title is not None:
        fig.suptitle(str(title), fontsize=13)

    fig.tight_layout(rect=[0, 0, 1, 0.95] if title is not None else None)
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return fig_path
