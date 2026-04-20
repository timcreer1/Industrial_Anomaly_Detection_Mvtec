"""SimCLR helpers for the MVTec anomaly detection project.

This module keeps together the logic that is specific to the self-supervised
pretraining stage. It focuses on:
- building the pooled normal-image SSL dataset and coverage summaries
- constructing the SimCLR encoder + projection-head model
- computing the NT-Xent contrastive loss from two augmented views
- training mild and strong SimCLR runs in a compact notebook-friendly way
- saving and loading encoder / full-model checkpoints plus run metadata

The shared transform logic stays in transform_utils.py and the shared backbone
loading logic stays in backbone_utils.py so this file can stay centred on the
SimCLR training stage.
"""

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision

try:
    from .transform_utils import build_simclr_transform
except Exception:
    from transform_utils import build_simclr_transform

try:
    from .backbone_utils import clean_state_dict_keys
except Exception:
    from backbone_utils import clean_state_dict_keys


#------------------------------------------------------------------------------
# Small general helpers
#------------------------------------------------------------------------------

# Resolve the torch device string used during training and scoring.
def resolve_device(device=None):
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device)


# Set all common random seeds for reproducible SSL runs.
def set_simclr_seed(seed=42, deterministic=False):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True


# Convert one object into a Path and create the parent directory when needed.
def ensure_parent_dir(file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


# Convert a torch scalar / tensor / NumPy value to a plain float for logging.
def to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


# Return a safe worker count for a runtime where CPU resources may vary.
def resolve_num_workers(num_workers=None):
    if num_workers is None:
        cpu_count = os.cpu_count() or 2
        return int(min(4, max(0, cpu_count // 2)))
    return int(max(0, num_workers))


# Return whether pinned memory should be used.
def resolve_pin_memory(pin_memory=None, device=None):
    if pin_memory is not None:
        return bool(pin_memory)
    return resolve_device(device) == "cuda"


# Return a torch autocast context for the current device.
def autocast_context(device=None, enabled=True):
    device = resolve_device(device)
    enabled = bool(enabled)

    if not enabled:
        return torch.autocast(device_type=device if device in ["cuda", "cpu"] else "cpu", enabled=False)

    if device == "cuda":
        return torch.autocast(device_type="cuda", enabled=True)

    # CPU autocast is optional and can be slower, so default to disabled there.
    return torch.autocast(device_type="cpu", enabled=False)


# Build a GradScaler that only activates on CUDA.
def build_grad_scaler(device=None, enabled=None):
    device = resolve_device(device)
    if enabled is None:
        enabled = (device == "cuda")

    try:
        return torch.amp.GradScaler(device, enabled=bool(enabled))
    except Exception:
        try:
            return torch.cuda.amp.GradScaler(enabled=bool(enabled and device == "cuda"))
        except Exception:
            return None


# Count the number of trainable parameters in one module.
def count_trainable_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# Return one checkpoint size in megabytes if the file exists.
def checkpoint_size_mb(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None
    return float(checkpoint_path.stat().st_size / (1024 ** 2))


#------------------------------------------------------------------------------
# SSL dataset and coverage helpers
#------------------------------------------------------------------------------

# Collect pooled train-good image paths from a split manifest or category dict.
def collect_ssl_paths(splits, categories=None, require_non_empty=True):
    if not isinstance(splits, dict):
        raise TypeError("splits must be a category-keyed dict.")

    if categories is None:
        categories = sorted(splits.keys())

    ssl_rows = []
    ssl_paths = []

    for category in categories:
        if category not in splits:
            raise KeyError(f"Category '{category}' not found in splits.")

        entry = splits[category]
        if not isinstance(entry, dict) or "train_good" not in entry:
            raise KeyError(
                f"Category '{category}' does not contain a 'train_good' list. "
                "Expected the split manifest produced by notebook 01."
            )

        train_good = [str(p) for p in entry["train_good"]]
        ssl_paths.extend(train_good)
        ssl_rows.append({"category": str(category), "n_train_good": int(len(train_good))})

    if require_non_empty and len(ssl_paths) == 0:
        raise RuntimeError("No SSL training images were found in the supplied splits.")

    df_coverage = pd.DataFrame(ssl_rows).sort_values("category").reset_index(drop=True)
    return ssl_paths, df_coverage


# Return a small one-row summary for the pooled SSL coverage.
def ssl_coverage_totals(df_coverage):
    if len(df_coverage) == 0:
        return {
            "n_categories": 0,
            "n_train_good_total": 0,
            "min_train_good": 0,
            "max_train_good": 0,
            "mean_train_good": 0.0,
        }

    counts = df_coverage["n_train_good"].astype(int)
    return {
        "n_categories": int(len(df_coverage)),
        "n_train_good_total": int(counts.sum()),
        "min_train_good": int(counts.min()),
        "max_train_good": int(counts.max()),
        "mean_train_good": float(counts.mean()),
    }


# Dataset that returns two augmented views of each normal training image.
class SimclrDataset(Dataset):
    def __init__(self, img_paths, transform, return_paths=False):
        self.img_paths = [str(p) for p in img_paths]
        self.transform = transform
        self.return_paths = bool(return_paths)

        if len(self.img_paths) == 0:
            raise ValueError("SimclrDataset received an empty img_paths list.")
        if self.transform is None:
            raise ValueError("SimclrDataset requires a valid transform.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path).convert("RGB")
        x1 = self.transform(img)
        x2 = self.transform(img)

        if self.return_paths:
            return x1, x2, img_path
        return x1, x2


# Build the SimCLR DataLoader from pooled normal training images.
def make_simclr_loader(
    img_paths,
    img_size=224,
    strength="mild",
    batch_size=128,
    shuffle=True,
    num_workers=None,
    pin_memory=None,
    drop_last=True,
    persistent_workers=None,
    prefetch_factor=None,
    transform=None,
    device=None,
    return_paths=False,
):
    num_workers = resolve_num_workers(num_workers)
    pin_memory = resolve_pin_memory(pin_memory=pin_memory, device=device)

    if transform is None:
        transform = build_simclr_transform(img_size=img_size, strength=strength)

    dataset = SimclrDataset(img_paths=img_paths, transform=transform, return_paths=return_paths)

    loader_kwargs = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "drop_last": bool(drop_last),
    }

    if num_workers > 0:
        if persistent_workers is None:
            persistent_workers = True
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(prefetch_factor)

    return DataLoader(dataset, **loader_kwargs)


#------------------------------------------------------------------------------
# Model definition
#------------------------------------------------------------------------------

# Build the plain ResNet-18 encoder used before the projection head is attached.
def get_resnet18_encoder():
    model = torchvision.models.resnet18(weights=None)
    feat_dim = int(model.fc.in_features)
    model.fc = nn.Identity()
    return model, feat_dim


# Projection head that maps encoder features into the contrastive space.
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), int(out_dim)),
        )

    def forward(self, x):
        return self.net(x)


# SimCLR model wrapper that combines the encoder and projection head.
class SimclrModel(nn.Module):
    def __init__(self, encoder, feat_dim, proj_dim=128, hidden_dim=512, normalize=True):
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionHead(in_dim=feat_dim, hidden_dim=hidden_dim, out_dim=proj_dim)
        self.normalize = bool(normalize)

    def encode(self, x, normalize=False):
        feats = self.encoder(x)
        if normalize:
            feats = F.normalize(feats, dim=1)
        return feats

    def project(self, feats, normalize=None):
        z = self.projector(feats)
        if normalize is None:
            normalize = self.normalize
        if normalize:
            z = F.normalize(z, dim=1)
        return z

    def forward(self, x):
        feats = self.encoder(x)
        z = self.projector(feats)
        if self.normalize:
            z = F.normalize(z, dim=1)
        return z


# Build the project-default SimCLR model.
def build_simclr_model(proj_dim=128, hidden_dim=512, device=None, normalize=True):
    device = resolve_device(device)
    encoder, feat_dim = get_resnet18_encoder()
    model = SimclrModel(
        encoder=encoder,
        feat_dim=feat_dim,
        proj_dim=proj_dim,
        hidden_dim=hidden_dim,
        normalize=normalize,
    )
    return model.to(device)


#------------------------------------------------------------------------------
# Loss helpers
#------------------------------------------------------------------------------

# Compute the standard SimCLR NT-Xent loss from two projected view batches.
def nt_xent_loss(z1, z2, temperature=0.2):
    if z1.ndim != 2 or z2.ndim != 2:
        raise ValueError("nt_xent_loss expects 2D tensors shaped [batch, feature_dim].")
    if z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have the same shape.")
    if z1.shape[0] < 2:
        raise ValueError("nt_xent_loss requires a batch size of at least 2.")

    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)

    sim = torch.matmul(z, z.T) / temperature
    eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)

    positive_index = torch.arange(2 * batch_size, device=sim.device)
    positive_index = (positive_index + batch_size) % (2 * batch_size)

    loss = F.cross_entropy(sim, positive_index)
    return loss


# Return one training step worth of loss and the current batch size.
def simclr_train_step(model, batch, optimizer, scaler=None, device=None, temperature=0.2, amp_enabled=None):
    device = resolve_device(device)
    if len(batch) < 2:
        raise ValueError("Expected a SimCLR batch containing at least two augmented views.")

    x1, x2 = batch[0], batch[1]
    x1 = x1.to(device, non_blocking=(device == "cuda"))
    x2 = x2.to(device, non_blocking=(device == "cuda"))

    if amp_enabled is None:
        amp_enabled = (device == "cuda")

    optimizer.zero_grad(set_to_none=True)

    with autocast_context(device=device, enabled=amp_enabled):
        z1 = model(x1)
        z2 = model(x2)
        loss = nt_xent_loss(z1, z2, temperature=temperature)

    if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

    return {
        "loss": to_float(loss),
        "batch_size": int(x1.shape[0]),
    }


#------------------------------------------------------------------------------
# Training helpers
#------------------------------------------------------------------------------

# Build the AdamW optimizer used for SimCLR pretraining.
def build_simclr_optimizer(model, lr=3e-4, weight_decay=1e-4):
    return torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))


# Run one SimCLR training epoch and return a compact summary dict.
def train_one_epoch_simclr(
    model,
    loader,
    optimizer,
    device=None,
    temperature=0.2,
    scaler=None,
    amp_enabled=None,
):
    device = resolve_device(device)
    model.train()

    running_loss = 0.0
    n_samples = 0
    n_steps = 0

    for batch in loader:
        step_out = simclr_train_step(
            model=model,
            batch=batch,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            temperature=temperature,
            amp_enabled=amp_enabled,
        )

        running_loss += float(step_out["loss"]) * int(step_out["batch_size"])
        n_samples += int(step_out["batch_size"])
        n_steps += 1

    if n_steps == 0 or n_samples == 0:
        raise RuntimeError("SimCLR loader produced no trainable batches.")

    return {
        "loss": float(running_loss / n_samples),
        "n_steps": int(n_steps),
        "n_samples": int(n_samples),
    }


# Convert one training-history list into a compact DataFrame.
def history_to_frame(history_rows):
    if len(history_rows) == 0:
        return pd.DataFrame(columns=["epoch", "loss", "sec_epoch", "n_steps", "n_samples"])
    return pd.DataFrame(history_rows)


# Fit one full SimCLR run and optionally save the best checkpoints and metadata.
def fit_simclr(
    img_paths,
    img_size=224,
    strength="mild",
    batch_size=128,
    epochs=30,
    lr=3e-4,
    weight_decay=1e-4,
    temperature=0.2,
    proj_dim=128,
    hidden_dim=512,
    device=None,
    num_workers=None,
    pin_memory=None,
    drop_last=True,
    persistent_workers=None,
    prefetch_factor=None,
    transform=None,
    seed=42,
    deterministic=False,
    save_dir=None,
    run_name=None,
    amp_enabled=None,
    verbose=True,
):
    device = resolve_device(device)
    set_simclr_seed(seed=seed, deterministic=deterministic)

    loader = make_simclr_loader(
        img_paths=img_paths,
        img_size=img_size,
        strength=strength,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        transform=transform,
        device=device,
        return_paths=False,
    )

    model = build_simclr_model(
        proj_dim=proj_dim,
        hidden_dim=hidden_dim,
        device=device,
        normalize=True,
    )

    optimizer = build_simclr_optimizer(model=model, lr=lr, weight_decay=weight_decay)
    if amp_enabled is None:
        amp_enabled = (device == "cuda")
    scaler = build_grad_scaler(device=device, enabled=amp_enabled)

    if run_name is None:
        run_name = f"simclr_{strength}"

    history_rows = []
    best_loss = math.inf
    best_epoch = None

    encoder_ckpt_path = None
    full_ckpt_path = None
    history_csv_path = None
    meta_json_path = None

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        encoder_ckpt_path = save_dir / f"{run_name}_encoder.pt"
        full_ckpt_path = save_dir / f"{run_name}_full.pt"
        history_csv_path = save_dir / f"{run_name}_history.csv"
        meta_json_path = save_dir / f"{run_name}_meta.json"

    train_t0 = time.time()

    for epoch in range(1, int(epochs) + 1):
        epoch_t0 = time.time()
        epoch_out = train_one_epoch_simclr(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            temperature=temperature,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )
        sec_epoch = float(time.time() - epoch_t0)

        row = {
            "epoch": int(epoch),
            "loss": float(epoch_out["loss"]),
            "sec_epoch": sec_epoch,
            "n_steps": int(epoch_out["n_steps"]),
            "n_samples": int(epoch_out["n_samples"]),
            "aug_strength": str(strength),
            "batch_size": int(batch_size),
            "temperature": float(temperature),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "proj_dim": int(proj_dim),
        }
        history_rows.append(row)

        if verbose:
            print(
                f"[SimCLR][{strength}] epoch {epoch:02d}/{int(epochs):02d} | "
                f"loss={row['loss']:.4f} | sec={sec_epoch:.1f}"
            )

        if row["loss"] < best_loss:
            best_loss = float(row["loss"])
            best_epoch = int(epoch)

            if save_dir is not None:
                save_simclr_checkpoint(
                    model=model,
                    encoder_path=encoder_ckpt_path,
                    full_model_path=full_ckpt_path,
                    epoch=epoch,
                    optimizer=optimizer,
                    extra_meta={
                        "aug_strength": str(strength),
                        "temperature": float(temperature),
                        "proj_dim": int(proj_dim),
                        "hidden_dim": int(hidden_dim),
                        "seed": int(seed),
                    },
                )

    total_sec = float(time.time() - train_t0)
    history_df = history_to_frame(history_rows)

    meta = build_simclr_run_metadata(
        run_name=run_name,
        aug_strength=strength,
        img_size=img_size,
        n_images=len(img_paths),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        temperature=temperature,
        proj_dim=proj_dim,
        hidden_dim=hidden_dim,
        seed=seed,
        deterministic=deterministic,
        device=device,
        best_loss=best_loss,
        best_epoch=best_epoch,
        total_sec=total_sec,
        encoder_checkpoint=encoder_ckpt_path,
        full_checkpoint=full_ckpt_path,
        n_params=count_trainable_params(model),
    )

    if save_dir is not None:
        history_df.to_csv(history_csv_path, index=False)
        save_json(meta, meta_json_path)

    return {
        "model": model,
        "history": history_rows,
        "history_df": history_df,
        "meta": meta,
        "loader": loader,
        "encoder_checkpoint": None if encoder_ckpt_path is None else str(encoder_ckpt_path),
        "full_checkpoint": None if full_ckpt_path is None else str(full_ckpt_path),
    }


#------------------------------------------------------------------------------
# Checkpoint helpers
#------------------------------------------------------------------------------

# Save the encoder-only and full-model SimCLR checkpoints.
def save_simclr_checkpoint(
    model,
    encoder_path=None,
    full_model_path=None,
    epoch=None,
    optimizer=None,
    extra_meta=None,
):
    if encoder_path is None and full_model_path is None:
        raise ValueError("At least one of encoder_path or full_model_path must be provided.")

    # Handle DataParallel wrappers cleanly.
    model_to_save = model.module if hasattr(model, "module") else model
    payload_common = {
        "epoch": None if epoch is None else int(epoch),
        "extra_meta": extra_meta or {},
    }

    if encoder_path is not None:
        encoder_path = ensure_parent_dir(encoder_path)
        encoder_payload = {
            **payload_common,
            "encoder_state_dict": model_to_save.encoder.state_dict(),
        }
        torch.save(encoder_payload, encoder_path)

    if full_model_path is not None:
        full_model_path = ensure_parent_dir(full_model_path)
        full_payload = {
            **payload_common,
            "state_dict": model_to_save.state_dict(),
        }
        if optimizer is not None:
            full_payload["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(full_payload, full_model_path)


# Load an encoder checkpoint into a plain ResNet-18 encoder.
def load_simclr_encoder(encoder_path, strict=True, map_location="cpu"):
    encoder_path = Path(encoder_path)
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder checkpoint not found: {encoder_path}")

    checkpoint_obj = torch.load(encoder_path, map_location=map_location)
    if isinstance(checkpoint_obj, dict) and "encoder_state_dict" in checkpoint_obj:
        state_dict = checkpoint_obj["encoder_state_dict"]
    elif isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
        state_dict = checkpoint_obj["state_dict"]
    else:
        state_dict = checkpoint_obj

    state_dict = clean_state_dict_keys(state_dict)
    encoder, feat_dim = get_resnet18_encoder()
    missing, unexpected = encoder.load_state_dict(state_dict, strict=strict)
    encoder.eval()

    return {
        "encoder": encoder,
        "feature_dim": int(feat_dim),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "checkpoint_path": str(encoder_path),
    }


# Load a full SimCLR model checkpoint when the projection head is also needed.
def load_simclr_model(full_model_path, proj_dim=128, hidden_dim=512, strict=True, map_location="cpu"):
    full_model_path = Path(full_model_path)
    if not full_model_path.exists():
        raise FileNotFoundError(f"Full SimCLR checkpoint not found: {full_model_path}")

    checkpoint_obj = torch.load(full_model_path, map_location=map_location)
    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj:
        state_dict = checkpoint_obj["state_dict"]
    else:
        state_dict = checkpoint_obj

    state_dict = clean_state_dict_keys(state_dict)
    model = build_simclr_model(proj_dim=proj_dim, hidden_dim=hidden_dim, device="cpu", normalize=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    model.eval()

    return {
        "model": model,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "checkpoint_path": str(full_model_path),
    }


#------------------------------------------------------------------------------
# Metadata / export helpers
#------------------------------------------------------------------------------

# Build one compact run metadata dict for JSON export.
def build_simclr_run_metadata(
    run_name,
    aug_strength,
    img_size,
    n_images,
    epochs,
    batch_size,
    lr,
    weight_decay,
    temperature,
    proj_dim,
    hidden_dim,
    seed,
    deterministic,
    device,
    best_loss,
    best_epoch,
    total_sec,
    encoder_checkpoint,
    full_checkpoint,
    n_params,
):
    return {
        "run_name": str(run_name),
        "aug_strength": str(aug_strength),
        "img_size": int(img_size),
        "n_images": int(n_images),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "temperature": float(temperature),
        "proj_dim": int(proj_dim),
        "hidden_dim": int(hidden_dim),
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "device": str(device),
        "best_loss": float(best_loss),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "total_sec": float(total_sec),
        "encoder_checkpoint": None if encoder_checkpoint is None else str(encoder_checkpoint),
        "full_checkpoint": None if full_checkpoint is None else str(full_checkpoint),
        "encoder_checkpoint_size_mb": None if encoder_checkpoint is None else checkpoint_size_mb(encoder_checkpoint),
        "full_checkpoint_size_mb": None if full_checkpoint is None else checkpoint_size_mb(full_checkpoint),
        "n_trainable_params": int(n_params),
    }


# Build one summary row suitable for the project CSV summary table.
def build_simclr_summary_row(meta):
    return {
        "run_name": meta.get("run_name"),
        "aug_strength": meta.get("aug_strength"),
        "img_size": meta.get("img_size"),
        "n_images": meta.get("n_images"),
        "epochs": meta.get("epochs"),
        "batch_size": meta.get("batch_size"),
        "lr": meta.get("lr"),
        "weight_decay": meta.get("weight_decay"),
        "temperature": meta.get("temperature"),
        "proj_dim": meta.get("proj_dim"),
        "hidden_dim": meta.get("hidden_dim"),
        "seed": meta.get("seed"),
        "device": meta.get("device"),
        "best_loss": meta.get("best_loss"),
        "best_epoch": meta.get("best_epoch"),
        "total_sec": meta.get("total_sec"),
        "encoder_checkpoint": meta.get("encoder_checkpoint"),
        "full_checkpoint": meta.get("full_checkpoint"),
        "encoder_checkpoint_size_mb": meta.get("encoder_checkpoint_size_mb"),
        "full_checkpoint_size_mb": meta.get("full_checkpoint_size_mb"),
        "n_trainable_params": meta.get("n_trainable_params"),
    }


# Save one object as JSON with a stable indentation.
def save_json(obj, json_path):
    json_path = ensure_parent_dir(json_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# Load one JSON object from disk.
def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Save the compact SSL coverage table and totals together.
def save_ssl_coverage_outputs(df_coverage, csv_path, json_path=None):
    csv_path = ensure_parent_dir(csv_path)
    df_coverage.to_csv(csv_path, index=False)

    if json_path is not None:
        totals = ssl_coverage_totals(df_coverage)
        save_json(totals, json_path)


# Save multiple run metas as one compact JSON file.
def save_simclr_runs_json(run_outputs, json_path):
    rows = []
    for item in run_outputs:
        if isinstance(item, dict) and "meta" in item:
            rows.append(item["meta"])
        elif isinstance(item, dict):
            rows.append(item)
    save_json(rows, json_path)

