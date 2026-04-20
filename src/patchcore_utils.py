"""PatchCore helpers for the MVTec anomaly detection project.

This module keeps together the logic that is specific to the PatchCore-style
nearest-neighbour baseline. It focuses on:
- collecting normal patch embeddings from one or more hooked backbone layers
- keeping a deterministic coreset subset of the training memory bank
- building a fast L2 nearest-neighbour index
- scoring images and producing anomaly heatmaps from patch distances
- exporting small metadata helpers used by the notebooks

The shared backbone / hook logic stays in backbone_utils.py so this file can stay
centred on PatchCore fitting and scoring.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .backbone_utils import forward_get_feats, concat_patch_features
except Exception:
    from backbone_utils import forward_get_feats, concat_patch_features

try:
    import faiss
    HAS_FAISS = True
except Exception:
    faiss = None
    HAS_FAISS = False


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


# Convert one array to contiguous float32 for indexing / searching.
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
# Search-index helpers
#------------------------------------------------------------------------------

# Small NumPy fallback index that mimics the FAISS search interface.
class NumpyL2Index:
    def __init__(self, bank_array):
        bank_array = as_float32_contiguous(bank_array)
        if bank_array.ndim != 2:
            raise ValueError("NumpyL2Index expects a 2D memory-bank array.")

        self.bank = bank_array
        self.ntotal = int(bank_array.shape[0])
        self.dim = int(bank_array.shape[1])

    # Return squared L2 distances and neighbour indices like FAISS.
    def search(self, queries, k=1):
        queries = as_float32_contiguous(queries)
        if queries.ndim != 2:
            raise ValueError("Queries must be a 2D array.")
        if queries.shape[1] != self.dim:
            raise ValueError(
                f"Query dim {queries.shape[1]} does not match index dim {self.dim}."
            )

        k = int(k)
        if k <= 0:
            raise ValueError("k must be at least 1.")
        k = min(k, self.ntotal)

        q_sq = np.sum(queries ** 2, axis=1, keepdims=True)
        b_sq = np.sum(self.bank ** 2, axis=1)[None, :]
        dist2 = q_sq + b_sq - 2.0 * (queries @ self.bank.T)
        dist2 = np.maximum(dist2, 0.0).astype(np.float32)

        if k == 1:
            idx = np.argmin(dist2, axis=1)[:, None]
            best = dist2[np.arange(dist2.shape[0]), idx[:, 0]][:, None]
            return best.astype(np.float32), idx.astype(np.int64)

        part_idx = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
        part_dist = np.take_along_axis(dist2, part_idx, axis=1)
        order = np.argsort(part_dist, axis=1)
        idx = np.take_along_axis(part_idx, order, axis=1)
        dist = np.take_along_axis(part_dist, order, axis=1)
        return dist.astype(np.float32), idx.astype(np.int64)


# Build the nearest-neighbour search index from a 2D memory-bank array.
def build_l2_index(bank_array, prefer_faiss=True):
    bank_array = as_float32_contiguous(bank_array)
    if bank_array.ndim != 2:
        raise ValueError("build_l2_index expects a 2D memory-bank array.")
    if bank_array.shape[0] == 0:
        raise ValueError("Cannot build an index from an empty memory bank.")

    if prefer_faiss and HAS_FAISS:
        index = faiss.IndexFlatL2(bank_array.shape[1])
        index.add(bank_array)
        return index

    return NumpyL2Index(bank_array)


# Backwards-compatible alias used in the notebooks.
def faiss_index_l2(array_2d, prefer_faiss=True):
    return build_l2_index(array_2d, prefer_faiss=prefer_faiss)


#------------------------------------------------------------------------------
# Memory-bank fitting helpers
#------------------------------------------------------------------------------

# Collect patch embeddings from normal training images.
def collect_patch_embeddings(
    model,
    features,
    train_loader,
    layer_names,
    device=None,
    max_train_patches=None,
):
    device = resolve_device(device)
    patch_chunks = []
    total_patches = 0

    for images, _, _, _ in train_loader:
        images = images.to(device, non_blocking=(device == "cuda"))
        feature_maps = forward_get_feats(model, features, images, layer_names)
        patches = concat_patch_features(feature_maps).detach().cpu().numpy()
        patches = patches.reshape(-1, patches.shape[-1]).astype(np.float32)
        patch_chunks.append(patches)
        total_patches += int(patches.shape[0])

        if max_train_patches is not None and total_patches >= int(max_train_patches):
            break

    if len(patch_chunks) == 0:
        raise RuntimeError("No training patch embeddings were collected for PatchCore.")

    patch_bank = np.concatenate(patch_chunks, axis=0).astype(np.float32)
    return patch_bank


# Deterministically cap a large patch bank to a maximum size.
def cap_patch_bank(bank_full, max_train_patches=None, seed=42, seed_label="patch_bank_cap"):
    bank_full = as_float32_contiguous(bank_full)

    if max_train_patches is None or len(bank_full) <= int(max_train_patches):
        return bank_full

    rng = np.random.default_rng(stable_seed_from_text(seed, seed_label))
    keep_idx = rng.choice(len(bank_full), size=int(max_train_patches), replace=False)
    return bank_full[keep_idx]


# Deterministically keep one random coreset subset from the full bank.
def select_patchcore_coreset(bank_full, coreset_ratio=0.05, seed=42, seed_label="patchcore_coreset"):
    bank_full = as_float32_contiguous(bank_full)
    if bank_full.ndim != 2:
        raise ValueError("select_patchcore_coreset expects a 2D bank array.")
    if len(bank_full) == 0:
        raise ValueError("The full patch bank is empty.")

    keep_n = max(1, int(round(len(bank_full) * float(coreset_ratio))))
    keep_n = min(keep_n, len(bank_full))

    rng = np.random.default_rng(stable_seed_from_text(seed, seed_label))
    keep_idx = rng.choice(len(bank_full), size=keep_n, replace=False)
    return bank_full[keep_idx], keep_idx.astype(np.int64)


# Build the final PatchCore memory bank from the normal training loader.
def build_patch_bank(
    model,
    features,
    train_loader,
    layer_names,
    category="global",
    device=None,
    coreset_ratio=0.05,
    max_train_patches=250000,
    seed=42,
):
    bank_full = collect_patch_embeddings(
        model=model,
        features=features,
        train_loader=train_loader,
        layer_names=layer_names,
        device=device,
        max_train_patches=max_train_patches,
    )

    bank_full = cap_patch_bank(
        bank_full,
        max_train_patches=max_train_patches,
        seed=seed,
        seed_label=f"{category}_patch_bank_cap",
    )

    bank, coreset_idx = select_patchcore_coreset(
        bank_full,
        coreset_ratio=coreset_ratio,
        seed=seed,
        seed_label=f"{category}_patchcore_coreset",
    )

    return bank, {
        "category": str(category),
        "n_bank_full": int(len(bank_full)),
        "n_bank_keep": int(len(bank)),
        "coreset_ratio": float(coreset_ratio),
        "max_train_patches": None if max_train_patches is None else int(max_train_patches),
        "coreset_idx": coreset_idx,
        "feature_dim": int(bank.shape[1]),
    }


# Fit a full PatchCore object that is ready for scoring.
def fit_patchcore(
    model,
    features,
    train_loader,
    layer_names,
    category="global",
    device=None,
    coreset_ratio=0.05,
    max_train_patches=250000,
    seed=42,
    prefer_faiss=True,
    out_size=None,
    patch_score_topk=64,
):
    layer_names = list(layer_names) if isinstance(layer_names, (list, tuple)) else [layer_names]
    device = resolve_device(device)

    bank, bank_meta = build_patch_bank(
        model=model,
        features=features,
        train_loader=train_loader,
        layer_names=layer_names,
        category=category,
        device=device,
        coreset_ratio=coreset_ratio,
        max_train_patches=max_train_patches,
        seed=seed,
    )

    index = build_l2_index(bank, prefer_faiss=prefer_faiss)

    patchcore_obj = {
        "memory_bank": bank,
        "index": index,
        "layer_names": layer_names,
        "device": device,
        "out_size": normalise_out_size(out_size),
        "patch_score_topk": int(patch_score_topk),
        "prefer_faiss": bool(prefer_faiss),
    }
    patchcore_obj.update(bank_meta)
    return patchcore_obj


#------------------------------------------------------------------------------
# Scoring helpers
#------------------------------------------------------------------------------

# Score one batch from nearest-neighbour patch distances and upsample the heatmap.
@torch.inference_mode()
def patchcore_scores(
    model,
    features,
    images,
    layer_names,
    index,
    device=None,
    out_size=None,
    patch_score_topk=64,
):
    device = resolve_device(device)
    images = images.to(device, non_blocking=(device == "cuda"))
    layer_names = list(layer_names) if isinstance(layer_names, (list, tuple)) else [layer_names]

    feature_maps = forward_get_feats(model, features, images, layer_names)
    patches = concat_patch_features(feature_maps).detach().cpu().numpy()

    batch_n, patch_n, feat_dim = patches.shape
    queries = patches.reshape(-1, feat_dim).astype(np.float32)
    dist2, _ = index.search(queries, 1)
    dist2 = dist2.reshape(batch_n, patch_n)

    feat_h, feat_w = feature_maps[-1].shape[-2:]
    heat = dist2.reshape(batch_n, feat_h, feat_w)

    target_size = normalise_out_size(out_size)
    if target_size is None:
        target_size = int(images.shape[-2]), int(images.shape[-1])

    heat_tensor = torch.from_numpy(heat).unsqueeze(1)
    heat_up = F.interpolate(
        heat_tensor,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(1).cpu().numpy()

    flat = heat_up.reshape(batch_n, -1)
    scores = topk_patch_score(flat, topk=patch_score_topk)
    return scores.astype(np.float32), heat_up.astype(np.float32)


# Convenience wrapper that scores from a fitted PatchCore object.
@torch.inference_mode()
def score_patchcore_batch(model, features, images, patchcore_obj):
    return patchcore_scores(
        model=model,
        features=features,
        images=images,
        layer_names=patchcore_obj["layer_names"],
        index=patchcore_obj["index"],
        device=patchcore_obj.get("device"),
        out_size=patchcore_obj.get("out_size"),
        patch_score_topk=patchcore_obj.get("patch_score_topk", 64),
    )


# Score every image in one loader and keep the main arrays for later metrics.
@torch.inference_mode()
def score_patchcore_loader(model, features, data_loader, patchcore_obj):
    y_true = []
    y_score = []
    masks = []
    heats = []
    paths = []

    for images, labels, mask_tensors, image_paths in data_loader:
        scores, heatmaps = score_patchcore_batch(model, features, images, patchcore_obj)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_score.extend(scores.tolist())
        masks.append(mask_tensors.detach().cpu().numpy())
        heats.append(heatmaps)
        paths.extend(list(image_paths))

    mask_array = np.concatenate(masks, axis=0) if len(masks) > 0 else np.zeros((0, 1, 1, 1), dtype=np.float32)
    heat_array = np.concatenate(heats, axis=0) if len(heats) > 0 else np.zeros((0, 1, 1), dtype=np.float32)

    return {
        "y_true": np.asarray(y_true, dtype=np.int64),
        "y_score": np.asarray(y_score, dtype=np.float32),
        "mask_array": mask_array.astype(np.float32),
        "heat_array": heat_array.astype(np.float32),
        "image_paths": paths,
    }


# Return the patch-level distances for one query matrix without reshaping.
def query_patchcore_index(index, query_patches, k=1):
    query_patches = as_float32_contiguous(query_patches)
    return index.search(query_patches, int(k))


#------------------------------------------------------------------------------
# Metadata helpers
#------------------------------------------------------------------------------

# Return a compact dictionary that can be logged alongside experiment outputs.
def patchcore_metadata(patchcore_obj):
    meta = {
        "category": patchcore_obj.get("category"),
        "layer_names": list(patchcore_obj.get("layer_names", [])),
        "device": patchcore_obj.get("device"),
        "n_bank_full": patchcore_obj.get("n_bank_full"),
        "n_bank_keep": patchcore_obj.get("n_bank_keep"),
        "coreset_ratio": patchcore_obj.get("coreset_ratio"),
        "feature_dim": patchcore_obj.get("feature_dim"),
        "patch_score_topk": patchcore_obj.get("patch_score_topk"),
        "prefer_faiss": patchcore_obj.get("prefer_faiss"),
        "out_size": patchcore_obj.get("out_size"),
    }
    return meta


# Estimate the memory-bank storage size in megabytes.
def patch_bank_size_mb(bank_array):
    bank_array = np.asarray(bank_array)
    return float(bank_array.nbytes / (1024 ** 2))


# Save only the memory-bank array so very small experiments can be reloaded.
def save_patch_bank(bank_array, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, as_float32_contiguous(bank_array))
    return save_path


# Load a saved memory-bank array from disk.
def load_patch_bank(save_path):
    save_path = Path(save_path)
    if not save_path.exists():
        raise FileNotFoundError(f"Patch bank file not found: {save_path}")
    return as_float32_contiguous(np.load(save_path))
