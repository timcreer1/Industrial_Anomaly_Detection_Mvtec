"""PaDiM helpers for the MVTec anomaly detection project.

This module keeps together the logic that is specific to the PaDiM-style
probabilistic baseline. It focuses on:
- collecting normal patch embeddings from one hooked backbone layer
- fitting location-wise Gaussian statistics from normal training patches
- optionally reducing the feature dimension deterministically
- scoring images with Mahalanobis distance maps
- exporting small metadata helpers used by the notebooks

The shared backbone / hook logic stays in backbone_utils.py so this file can stay
centred on PaDiM fitting and scoring.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .backbone_utils import forward_get_feats
except Exception:
    from backbone_utils import forward_get_feats


#------------------------------------------------------------------------------
# Small general helpers
#------------------------------------------------------------------------------

# Create a deterministic integer seed from a base seed and text label.
def stable_seed_from_text(base_seed, text):
    base_seed = int(base_seed)
    text = str(text)
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return int((base_seed + int(digest, 16)) % (2 ** 32 - 1))


# Resolve the torch device string used during forward passes.
def resolve_device(device=None):
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


# Convert one array to contiguous float32.
def as_float32_contiguous(array_like):
    return np.ascontiguousarray(np.asarray(array_like, dtype=np.float32))


# Return a clean (height, width) pair from various size inputs.
def normalise_out_size(out_size):
    if out_size is None:
        return None

    if isinstance(out_size, int):
        return int(out_size), int(out_size)

    if isinstance(out_size, (tuple, list)) and len(out_size) == 2:
        return int(out_size[0]), int(out_size[1])

    raise ValueError("out_size must be None, an int, or a (height, width) pair.")


# Compute the top-k mean score from one 2D flattened anomaly map array.
def topk_patch_score(flat_scores, topk=64):
    flat_scores = np.asarray(flat_scores, dtype=np.float32)
    if flat_scores.ndim != 2:
        raise ValueError("topk_patch_score expects a 2D array shaped [batch, n_pixels].")

    topk_n = min(int(topk), flat_scores.shape[1])
    if topk_n <= 0:
        raise ValueError("topk must resolve to at least 1 pixel.")

    return np.mean(np.sort(flat_scores, axis=1)[:, -topk_n:], axis=1)


#------------------------------------------------------------------------------
# Feature collection helpers
#------------------------------------------------------------------------------

# Deterministically choose which feature channels to keep for PaDiM.
def select_padim_dims(total_dim, padim_dim=None, seed=42, seed_label="padim_dims"):
    total_dim = int(total_dim)
    if total_dim <= 0:
        raise ValueError("total_dim must be positive.")

    if padim_dim is None:
        keep_dim = total_dim
    else:
        keep_dim = min(int(padim_dim), total_dim)

    if keep_dim <= 0:
        raise ValueError("padim_dim must resolve to at least 1 channel.")

    if keep_dim == total_dim:
        return np.arange(total_dim, dtype=np.int64)

    rng = np.random.default_rng(stable_seed_from_text(seed, seed_label))
    keep_idx = rng.choice(total_dim, size=keep_dim, replace=False)
    keep_idx = np.sort(keep_idx).astype(np.int64)
    return keep_idx


# Collect one batchable matrix shaped [n_images, n_patches, n_channels].
def collect_padim_embeddings(model, features, train_loader, layer_name, device=None):
    device = resolve_device(device)
    chunks = []

    for images, _, _, _ in train_loader:
        images = images.to(device, non_blocking=(device == "cuda"))
        feature_map = forward_get_feats(model, features, images, [layer_name])[0]
        batch_np = feature_map.detach().cpu().numpy().astype(np.float32)
        b, c, h, w = batch_np.shape
        batch_np = batch_np.transpose(0, 2, 3, 1).reshape(b, h * w, c)
        chunks.append(batch_np)

    if len(chunks) == 0:
        raise RuntimeError("No training feature embeddings were collected for PaDiM.")

    embeddings = np.concatenate(chunks, axis=0).astype(np.float32)
    return embeddings


#------------------------------------------------------------------------------
# Gaussian fitting helpers
#------------------------------------------------------------------------------

# Fit one Gaussian per spatial location using the normal training embeddings.
def fit_gaussian_stats(embeddings, eps=0.01):
    embeddings = as_float32_contiguous(embeddings)
    if embeddings.ndim != 3:
        raise ValueError("fit_gaussian_stats expects [n_images, n_patches, n_channels].")

    n_images, n_patches, n_channels = embeddings.shape
    if n_images < 2:
        raise ValueError("PaDiM requires at least 2 normal images to estimate covariance.")

    mu = np.zeros((n_patches, n_channels), dtype=np.float32)
    cov_inv = np.zeros((n_patches, n_channels, n_channels), dtype=np.float32)

    eye = np.eye(n_channels, dtype=np.float32)

    for patch_idx in range(n_patches):
        x = embeddings[:, patch_idx, :]
        mu_patch = x.mean(axis=0).astype(np.float32)
        xc = x - mu_patch[None, :]
        cov = (xc.T @ xc) / max(1, n_images - 1)
        cov = cov.astype(np.float32) + (float(eps) * eye)
        cov_inv_patch = np.linalg.inv(cov).astype(np.float32)

        mu[patch_idx] = mu_patch
        cov_inv[patch_idx] = cov_inv_patch

    return mu, cov_inv


# Fit the full PaDiM statistics object from the normal training loader.
def fit_padim(
    model,
    features,
    train_loader,
    layer_name,
    category="global",
    device=None,
    padim_dim=100,
    eps=0.01,
    seed=42,
    out_size=None,
    patch_score_topk=64,
):
    device = resolve_device(device)
    embeddings_full = collect_padim_embeddings(
        model=model,
        features=features,
        train_loader=train_loader,
        layer_name=layer_name,
        device=device,
    )

    n_train, n_patches, full_dim = embeddings_full.shape
    keep_idx = select_padim_dims(
        total_dim=full_dim,
        padim_dim=padim_dim,
        seed=seed,
        seed_label=f"{category}_{layer_name}_padim_dims",
    )

    embeddings = embeddings_full[:, :, keep_idx]
    mu, cov_inv = fit_gaussian_stats(embeddings, eps=eps)

    stats = {
        "category": str(category),
        "layer_name": str(layer_name),
        "mu": mu,
        "cov_inv": cov_inv,
        "keep_idx": keep_idx.astype(np.int64),
        "grid_h": int(round(np.sqrt(n_patches))),
        "grid_w": int(n_patches / max(1, int(round(np.sqrt(n_patches))))),
        "n_train": int(n_train),
        "n_patches": int(n_patches),
        "feature_dim_full": int(full_dim),
        "feature_dim_keep": int(len(keep_idx)),
        "eps": float(eps),
        "seed": int(seed),
        "out_size": normalise_out_size(out_size),
        "patch_score_topk": int(patch_score_topk),
    }

    return stats


#------------------------------------------------------------------------------
# Scoring helpers
#------------------------------------------------------------------------------

# Compute Mahalanobis maps from one 3D embedding tensor and fitted stats.
def mahalanobis_maps(embeddings, mu, cov_inv):
    embeddings = as_float32_contiguous(embeddings)
    mu = as_float32_contiguous(mu)
    cov_inv = as_float32_contiguous(cov_inv)

    if embeddings.ndim != 3:
        raise ValueError("embeddings must be [batch, n_patches, n_channels].")
    if mu.ndim != 2:
        raise ValueError("mu must be [n_patches, n_channels].")
    if cov_inv.ndim != 3:
        raise ValueError("cov_inv must be [n_patches, n_channels, n_channels].")

    if embeddings.shape[1] != mu.shape[0] or embeddings.shape[2] != mu.shape[1]:
        raise ValueError("Embedding shape does not match fitted PaDiM statistics.")

    diff = embeddings - mu[None, :, :]
    # einsum over batch, patch, channel dims.
    dist2 = np.einsum("bpc,pcd,bpd->bp", diff, cov_inv, diff, optimize=True)
    dist2 = np.maximum(dist2, 0.0).astype(np.float32)
    return dist2


# Score one batch and upsample the PaDiM heatmaps to image size.
@torch.no_grad()
def padim_scores(model, features, x, layer_name, stats, device=None, out_size=None, patch_score_topk=None):
    device = resolve_device(device)
    x = x.to(device, non_blocking=(device == "cuda"))

    feature_map = forward_get_feats(model, features, x, [layer_name])[0]
    fmap_np = feature_map.detach().cpu().numpy().astype(np.float32)
    b, c, h, w = fmap_np.shape
    patches = fmap_np.transpose(0, 2, 3, 1).reshape(b, h * w, c)

    keep_idx = np.asarray(stats["keep_idx"], dtype=np.int64)
    patches = patches[:, :, keep_idx]

    dist2 = mahalanobis_maps(patches, stats["mu"], stats["cov_inv"])
    heat = dist2.reshape(b, h, w).astype(np.float32)

    target_size = normalise_out_size(out_size)
    if target_size is None:
        target_size = stats.get("out_size")
    if target_size is None:
        target_size = tuple(int(v) for v in x.shape[-2:])

    heat_t = torch.from_numpy(heat).unsqueeze(1)
    heat_up = F.interpolate(
        heat_t,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1).cpu().numpy().astype(np.float32)

    topk_val = patch_score_topk
    if topk_val is None:
        topk_val = int(stats.get("patch_score_topk", 64))

    flat = heat_up.reshape(b, -1)
    scores = topk_patch_score(flat, topk=topk_val)
    return scores.astype(np.float32), heat_up.astype(np.float32)


# Score a full dataloader and return stacked arrays plus per-item metadata.
@torch.no_grad()
def score_padim_loader(model, features, test_loader, layer_name, stats, device=None, out_size=None, patch_score_topk=None):
    labels_all = []
    scores_all = []
    heat_all = []
    masks_all = []
    paths_all = []

    for images, labels, masks, paths in test_loader:
        scores, heatmaps = padim_scores(
            model=model,
            features=features,
            x=images,
            layer_name=layer_name,
            stats=stats,
            device=device,
            out_size=out_size,
            patch_score_topk=patch_score_topk,
        )
        labels_all.append(np.asarray(labels, dtype=np.int64))
        scores_all.append(np.asarray(scores, dtype=np.float32))
        heat_all.append(np.asarray(heatmaps, dtype=np.float32))
        masks_all.append(np.asarray(masks, dtype=np.float32))
        paths_all.extend(list(paths))

    if len(scores_all) == 0:
        raise RuntimeError("The PaDiM test loader produced no batches.")

    result = {
        "labels": np.concatenate(labels_all, axis=0).astype(np.int64),
        "scores": np.concatenate(scores_all, axis=0).astype(np.float32),
        "heatmaps": np.concatenate(heat_all, axis=0).astype(np.float32),
        "masks": np.concatenate(masks_all, axis=0).astype(np.float32),
        "paths": paths_all,
    }
    return result


#------------------------------------------------------------------------------
# Metadata helpers
#------------------------------------------------------------------------------

# Return a compact metadata row for one fitted PaDiM object.
def padim_metadata_row(stats):
    bytes_total = int(stats["mu"].nbytes + stats["cov_inv"].nbytes + stats["keep_idx"].nbytes)
    return {
        "category": str(stats.get("category", "unknown")),
        "layer_name": str(stats.get("layer_name", "unknown")),
        "n_train": int(stats["n_train"]),
        "n_patches": int(stats["n_patches"]),
        "feature_dim_full": int(stats["feature_dim_full"]),
        "feature_dim_keep": int(stats["feature_dim_keep"]),
        "eps": float(stats["eps"]),
        "stats_mb": float(bytes_total / (1024 ** 2)),
        "grid_h": int(stats["grid_h"]),
        "grid_w": int(stats["grid_w"]),
        "patch_score_topk": int(stats.get("patch_score_topk", 64)),
    }


# Save one fitted PaDiM statistics object as a compressed .npz file.
def save_padim_stats(stats, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        mu=as_float32_contiguous(stats["mu"]),
        cov_inv=as_float32_contiguous(stats["cov_inv"]),
        keep_idx=np.asarray(stats["keep_idx"], dtype=np.int64),
        category=np.asarray(str(stats.get("category", "global"))),
        layer_name=np.asarray(str(stats.get("layer_name", "layer3"))),
        n_train=np.asarray(int(stats.get("n_train", 0))),
        n_patches=np.asarray(int(stats.get("n_patches", 0))),
        feature_dim_full=np.asarray(int(stats.get("feature_dim_full", 0))),
        feature_dim_keep=np.asarray(int(stats.get("feature_dim_keep", 0))),
        eps=np.asarray(float(stats.get("eps", 0.01))),
        seed=np.asarray(int(stats.get("seed", 42))),
        patch_score_topk=np.asarray(int(stats.get("patch_score_topk", 64))),
    )
    return out_path


# Load one fitted PaDiM statistics object from a compressed .npz file.
def load_padim_stats(npz_path):
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"PaDiM stats file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        stats = {
            "mu": as_float32_contiguous(data["mu"]),
            "cov_inv": as_float32_contiguous(data["cov_inv"]),
            "keep_idx": np.asarray(data["keep_idx"], dtype=np.int64),
            "category": str(data["category"].item()),
            "layer_name": str(data["layer_name"].item()),
            "n_train": int(data["n_train"].item()),
            "n_patches": int(data["n_patches"].item()),
            "feature_dim_full": int(data["feature_dim_full"].item()),
            "feature_dim_keep": int(data["feature_dim_keep"].item()),
            "eps": float(data["eps"].item()),
            "seed": int(data["seed"].item()) if "seed" in data else 42,
            "patch_score_topk": int(data["patch_score_topk"].item()) if "patch_score_topk" in data else 64,
        }

    n_patches = int(stats["n_patches"])
    grid_h = int(round(np.sqrt(n_patches)))
    grid_h = max(1, grid_h)
    grid_w = max(1, n_patches // grid_h)
    stats["grid_h"] = grid_h
    stats["grid_w"] = grid_w
    stats["out_size"] = None
    return stats
