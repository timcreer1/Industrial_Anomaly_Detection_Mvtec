"""Autoencoder helpers for the MVTec anomaly detection project.

This module keeps together the logic that is specific to the reconstruction
baseline. It focuses on:
- defining a lightweight convolutional autoencoder used in the project
- training the model on normal images only
- scoring images through reconstruction error heatmaps
- evaluating loaders and exporting compact training metadata
- saving and loading checkpoints used across notebooks

The goal is to keep the reconstruction baseline self-contained so the notebooks
can stay cleaner and more consistent with the PatchCore and PaDiM utilities.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


#------------------------------------------------------------------------------
# Small general helpers
#------------------------------------------------------------------------------

# Resolve the torch device string used during training and scoring.
def resolve_device(device=None):
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


# Convert one object into a Path and create the parent directory when needed.
def ensure_parent_dir(file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


# Return a clean (height, width) pair from various size inputs.
def normalise_out_size(out_size):
    if out_size is None:
        return None

    if isinstance(out_size, int):
        return int(out_size), int(out_size)

    if isinstance(out_size, (tuple, list)) and len(out_size) == 2:
        return int(out_size[0]), int(out_size[1])

    raise ValueError("out_size must be None, an int, or a (height, width) pair.")


# Convert a torch scalar / tensor / NumPy value to a plain float for logging.
def to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


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
# Model definition
#------------------------------------------------------------------------------

# Small convolutional autoencoder used as the reconstruction baseline.
class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels=3, latent_channels=256):
        super().__init__()

        latent_channels = int(latent_channels)
        if latent_channels <= 0:
            raise ValueError("latent_channels must be positive.")

        # 224 -> 112 -> 56 -> 28 -> 14
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # 14 -> 28 -> 56 -> 112 -> 224
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


# Build the default autoencoder for the notebooks.
def build_autoencoder(in_channels=3, latent_channels=256, device=None):
    device = resolve_device(device)
    model = ConvAutoencoder(in_channels=in_channels, latent_channels=latent_channels)
    return model.to(device)


#------------------------------------------------------------------------------
# Loss and scoring helpers
#------------------------------------------------------------------------------

# Return the element-wise reconstruction loss map for one batch.
def reconstruction_error_map(x, recon, reduction="mean_channel"):
    if x.shape != recon.shape:
        raise ValueError("x and recon must have the same shape.")

    err = (x - recon) ** 2

    if reduction == "none":
        return err
    if reduction == "mean_channel":
        return err.mean(dim=1, keepdim=True)
    if reduction == "sum_channel":
        return err.sum(dim=1, keepdim=True)

    raise ValueError("reduction must be one of: none, mean_channel, sum_channel.")


# Reduce a 4D reconstruction error map to one scalar score per image.
def reduce_error_map_to_scores(error_map, score_mode="topk_mean", topk=64):
    if not isinstance(error_map, torch.Tensor):
        raise TypeError("error_map must be a torch.Tensor.")
    if error_map.ndim != 4:
        raise ValueError("error_map must be [batch, channels, height, width].")

    flat = error_map.detach().cpu().numpy().reshape(error_map.shape[0], -1).astype(np.float32)

    if score_mode == "mean":
        return flat.mean(axis=1)
    if score_mode == "max":
        return flat.max(axis=1)
    if score_mode == "topk_mean":
        return topk_patch_score(flat, topk=topk)

    raise ValueError("score_mode must be one of: mean, max, topk_mean.")


# Run the autoencoder on one batch and return reconstructions, heatmaps, and scores.
@torch.no_grad()
def autoencoder_scores(model, x, device=None, score_mode="topk_mean", topk=64, out_size=None):
    device = resolve_device(device)
    x = x.to(device, non_blocking=(device == "cuda"))

    recon = model(x)
    heat = reconstruction_error_map(x, recon, reduction="mean_channel")

    target_size = normalise_out_size(out_size)
    if target_size is not None and tuple(heat.shape[-2:]) != tuple(target_size):
        heat = F.interpolate(heat, size=target_size, mode="bilinear", align_corners=False)

    scores = reduce_error_map_to_scores(heat, score_mode=score_mode, topk=topk)
    heat_np = heat[:, 0].detach().cpu().numpy().astype(np.float32)
    recon_np = recon.detach().cpu().numpy().astype(np.float32)

    return {
        "recon": recon_np,
        "heatmaps": heat_np,
        "scores": np.asarray(scores, dtype=np.float32),
    }


# Score one loader and return image-level predictions plus stored arrays.
@torch.no_grad()
def score_autoencoder_loader(
    model,
    loader,
    device=None,
    score_mode="topk_mean",
    topk=64,
    out_size=None,
):
    device = resolve_device(device)

    all_scores = []
    all_labels = []
    all_categories = []
    all_paths = []
    all_heatmaps = []

    for images, labels, masks, meta in loader:
        batch_out = autoencoder_scores(
            model=model,
            x=images,
            device=device,
            score_mode=score_mode,
            topk=topk,
            out_size=out_size,
        )

        batch_n = len(batch_out["scores"])
        all_scores.extend(batch_out["scores"].tolist())
        all_labels.extend(np.asarray(labels).reshape(-1).astype(int).tolist())
        all_heatmaps.extend(list(batch_out["heatmaps"]))

        if isinstance(meta, dict):
            all_categories.extend(list(meta.get("category", [""] * batch_n)))
            all_paths.extend(list(meta.get("img_path", [""] * batch_n)))
        else:
            all_categories.extend([""] * batch_n)
            all_paths.extend([""] * batch_n)

    df = pd.DataFrame({
        "category": all_categories,
        "img_path": all_paths,
        "label": all_labels,
        "image_score": all_scores,
    })

    return {
        "table": df,
        "heatmaps": np.asarray(all_heatmaps, dtype=np.float32),
        "scores": np.asarray(all_scores, dtype=np.float32),
        "labels": np.asarray(all_labels, dtype=np.int64),
    }


#------------------------------------------------------------------------------
# Training helpers
#------------------------------------------------------------------------------

# Build the optimizer used for the reconstruction baseline.
def build_autoencoder_optimizer(model, lr=1e-3, weight_decay=0.0):
    return torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))


# Run one training epoch on normal images only.
def train_one_epoch_autoencoder(model, loader, optimizer, device=None):
    device = resolve_device(device)
    model.train()

    running_loss = 0.0
    n_images = 0

    for images, _, _, _ in loader:
        images = images.to(device, non_blocking=(device == "cuda"))

        optimizer.zero_grad(set_to_none=True)
        recon = model(images)
        loss = F.mse_loss(recon, images, reduction="mean")
        loss.backward()
        optimizer.step()

        batch_n = images.size(0)
        running_loss += to_float(loss) * batch_n
        n_images += batch_n

    if n_images == 0:
        raise RuntimeError("No images were seen during autoencoder training.")

    return {
        "train_loss": float(running_loss / n_images),
        "n_train_images": int(n_images),
    }


# Evaluate one loader with the mean reconstruction loss.
@torch.no_grad()
def evaluate_autoencoder_recon_loss(model, loader, device=None):
    device = resolve_device(device)
    model.eval()

    running_loss = 0.0
    n_images = 0

    for images, _, _, _ in loader:
        images = images.to(device, non_blocking=(device == "cuda"))
        recon = model(images)
        loss = F.mse_loss(recon, images, reduction="mean")

        batch_n = images.size(0)
        running_loss += to_float(loss) * batch_n
        n_images += batch_n

    if n_images == 0:
        raise RuntimeError("No images were seen during autoencoder evaluation.")

    return {
        "recon_loss": float(running_loss / n_images),
        "n_eval_images": int(n_images),
    }


# Train the autoencoder for multiple epochs and return history plus the best model state.
def fit_autoencoder(
    model,
    train_loader,
    val_loader=None,
    device=None,
    epochs=5,
    lr=1e-3,
    weight_decay=0.0,
    score_mode="topk_mean",
    topk=64,
    out_size=None,
):
    device = resolve_device(device)
    model = model.to(device)
    optimizer = build_autoencoder_optimizer(model, lr=lr, weight_decay=weight_decay)

    history_rows = []
    best_metric = None
    best_epoch = None
    best_state = None

    for epoch in range(1, int(epochs) + 1):
        train_stats = train_one_epoch_autoencoder(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        row = {
            "epoch": int(epoch),
            "train_loss": float(train_stats["train_loss"]),
            "n_train_images": int(train_stats["n_train_images"]),
        }

        if val_loader is not None:
            val_stats = evaluate_autoencoder_recon_loss(model, val_loader, device=device)
            row["val_recon_loss"] = float(val_stats["recon_loss"])
            row["n_val_images"] = int(val_stats["n_eval_images"])
            current_metric = row["val_recon_loss"]
        else:
            current_metric = row["train_loss"]

        history_rows.append(row)

        if best_metric is None or current_metric < best_metric:
            best_metric = float(current_metric)
            best_epoch = int(epoch)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Autoencoder training did not produce a valid best checkpoint.")

    model.load_state_dict(best_state)

    history_df = pd.DataFrame(history_rows)
    summary = {
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "epochs": int(epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "score_mode": str(score_mode),
        "topk": int(topk),
        "out_size": normalise_out_size(out_size),
    }

    return {
        "model": model,
        "history": history_df,
        "summary": summary,
        "best_state_dict": best_state,
    }


#------------------------------------------------------------------------------
# Metadata and export helpers
#------------------------------------------------------------------------------

# Build one compact metadata dictionary for the trained autoencoder.
def build_autoencoder_metadata(
    category="global",
    latent_channels=256,
    epochs=5,
    best_epoch=None,
    best_metric=None,
    train_rows=None,
    val_rows=None,
    img_size=224,
    score_mode="topk_mean",
    topk=64,
):
    return {
        "category": str(category),
        "latent_channels": int(latent_channels),
        "epochs": int(epochs),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "best_metric": None if best_metric is None else float(best_metric),
        "train_rows": None if train_rows is None else int(train_rows),
        "val_rows": None if val_rows is None else int(val_rows),
        "img_size": int(img_size),
        "score_mode": str(score_mode),
        "topk": int(topk),
    }


# Save a trained autoencoder checkpoint with optional metadata and history.
def save_autoencoder_checkpoint(model, checkpoint_path, metadata=None, history_df=None):
    checkpoint_path = ensure_parent_dir(checkpoint_path)

    payload = {
        "state_dict": model.state_dict(),
        "metadata": {} if metadata is None else dict(metadata),
    }

    if history_df is not None:
        payload["history"] = history_df.to_dict(orient="records")

    torch.save(payload, checkpoint_path)
    return checkpoint_path


# Load a trained autoencoder checkpoint.
def load_autoencoder_checkpoint(
    checkpoint_path,
    model=None,
    device=None,
    strict=True,
    map_location="cpu",
):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=map_location)
    state_dict = payload.get("state_dict", payload)

    if model is None:
        metadata = payload.get("metadata", {})
        latent_channels = int(metadata.get("latent_channels", 256))
        model = build_autoencoder(latent_channels=latent_channels, device=device)
    else:
        model = model.to(resolve_device(device))

    model.load_state_dict(state_dict, strict=bool(strict))
    model.eval()

    history_df = None
    if "history" in payload:
        history_df = pd.DataFrame(payload["history"])

    return {
        "model": model,
        "metadata": payload.get("metadata", {}),
        "history": history_df,
        "payload": payload,
    }


# Save the training history to CSV.
def save_autoencoder_history(history_df, csv_path):
    csv_path = ensure_parent_dir(csv_path)
    history_df.to_csv(csv_path, index=False)
    return csv_path


# Save metadata to JSON.
def save_autoencoder_metadata(metadata, json_path):
    json_path = ensure_parent_dir(json_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return json_path


# Load metadata from JSON.
def load_autoencoder_metadata(json_path):
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
