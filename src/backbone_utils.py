"""Shared backbone and feature-hook utilities for the MVTec project.

This module keeps together the backbone-related helpers that are reused across
multiple notebooks. It focuses on:
- building the ImageNet ResNet-18 backbone used by PatchCore and PaDiM
- building the ResNet-18 encoder shape used by SimCLR checkpoints
- registering and clearing intermediate feature hooks
- converting feature maps into patch matrices for downstream anomaly scoring
- loading saved encoder checkpoints safely, including DataParallel prefixes

Model-specific fitting logic should stay in patchcore_utils.py, padim_utils.py,
and simclr_utils.py so this file stays centred on shared backbone handling.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


#------------------------------------------------------------------------------
# Backbone builders
#------------------------------------------------------------------------------

# Build the standard ImageNet-pretrained ResNet-18 used by the feature methods.
def get_resnet_imagenet(weights="IMAGENET1K_V1", strict_availability=True):
    weights_obj = None

    if weights is not None:
        try:
            if isinstance(weights, str):
                weights_obj = getattr(torchvision.models.ResNet18_Weights, weights)
            else:
                weights_obj = weights
        except AttributeError as exc:
            raise ValueError(
                f"Unknown ResNet18 weight setting: {weights}. "
                "Use None or a valid torchvision.models.ResNet18_Weights entry."
            ) from exc

    try:
        model = torchvision.models.resnet18(weights=weights_obj)
    except Exception as exc:
        if weights_obj is None or not strict_availability:
            model = torchvision.models.resnet18(weights=None)
        else:
            raise RuntimeError(
                "Could not load the requested ImageNet weights for ResNet-18. "
                "This can happen in offline runtimes that do not already have the "
                "weights cached. Either pre-cache the weights or call "
                "get_resnet_imagenet(weights=None)."
            ) from exc

    # Remove the classification head because downstream methods use feature maps.
    model.fc = nn.Identity()
    model.eval()
    return model


# Build the same ResNet-18 encoder shape used for SimCLR checkpoints.
def get_resnet18_ssl():
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Identity()
    model.eval()
    return model


# Small convenience wrapper for the common project backbone names.
def build_backbone(backbone_name="resnet18_imagenet", strict_availability=True):
    backbone_name = str(backbone_name).strip().lower()

    if backbone_name in ["resnet18_imagenet", "imagenet", "resnet18"]:
        return get_resnet_imagenet(strict_availability=strict_availability)

    if backbone_name in ["resnet18_ssl", "ssl", "simclr", "resnet18_random"]:
        return get_resnet18_ssl()

    raise ValueError(f"Unknown backbone_name: {backbone_name}")


#------------------------------------------------------------------------------
# Layer access helpers
#------------------------------------------------------------------------------

# Return one nested module from a dotted layer path such as "layer3".
def get_module_by_name(model, layer_name):
    layer_name = str(layer_name)
    module = model

    for part in layer_name.split("."):
        if not hasattr(module, part):
            available = list_named_hook_layers(model)
            raise AttributeError(
                f"Model does not have layer '{layer_name}'. "
                f"Closest available top-level hook layers: {available}"
            )
        module = getattr(module, part)

    return module


# Return the list of requested layer names after validating that they exist.
def validate_layer_names(model, layer_names):
    if isinstance(layer_names, str):
        layer_names = [layer_names]

    validated = []
    seen = set()

    for name in layer_names:
        name = str(name).strip()
        if name == "":
            continue
        _ = get_module_by_name(model, name)
        if name not in seen:
            validated.append(name)
            seen.add(name)

    if len(validated) == 0:
        raise ValueError("No valid layer names were supplied.")

    return validated


# Return the common hookable layers for a ResNet-like backbone.
def list_named_hook_layers(model):
    preferred = [
        "conv1", "bn1", "relu", "maxpool",
        "layer1", "layer2", "layer3", "layer4",
        "avgpool", "fc",
    ]
    return [name for name in preferred if hasattr(model, name)]


#------------------------------------------------------------------------------
# Hook registration helpers
#------------------------------------------------------------------------------

# Register hooks so intermediate feature maps can be captured during a forward pass.
def make_feature_hooks(model, layer_names):
    layer_names = validate_layer_names(model, layer_names)
    features = {}
    handles = []

    def hook_fn(name):
        def _hook(_, __, output):
            features[name] = output
        return _hook

    for layer_name in layer_names:
        module = get_module_by_name(model, layer_name)
        handle = module.register_forward_hook(hook_fn(layer_name))
        handles.append(handle)

    return features, handles


# Remove hook handles cleanly so repeated notebook runs do not stack hooks.
def remove_hook_handles(handles):
    for handle in handles:
        try:
            handle.remove()
        except Exception:
            pass


# Clear the saved feature dictionary in-place between forwards if desired.
def clear_feature_cache(features):
    if isinstance(features, dict):
        features.clear()


# Simple context-manager wrapper so hooks can be cleaned up automatically.
class FeatureHookContext:
    def __init__(self, model, layer_names):
        self.model = model
        self.layer_names = layer_names
        self.features = None
        self.handles = None

    def __enter__(self):
        self.features, self.handles = make_feature_hooks(self.model, self.layer_names)
        return self.features, self.handles

    def __exit__(self, exc_type, exc_val, exc_tb):
        remove_hook_handles(self.handles or [])
        return False


#------------------------------------------------------------------------------
# Forward and patch helpers
#------------------------------------------------------------------------------

# Run one forward pass and return the requested feature maps in the requested order.
@torch.no_grad()
def forward_get_feats(model, features, x, layer_names, clear_before=False):
    layer_names = validate_layer_names(model, layer_names)

    if clear_before:
        clear_feature_cache(features)

    _ = model(x)
    missing = [name for name in layer_names if name not in features]
    if len(missing) > 0:
        raise RuntimeError(
            f"Feature hooks did not capture layers: {missing}. "
            "Check that hooks were registered before calling forward_get_feats()."
        )

    return [features[name] for name in layer_names]


# Flatten one BCHW feature map into a batch of patch-by-feature matrices.
def fmap_to_patches(feature_map):
    if feature_map.ndim != 4:
        raise ValueError("fmap_to_patches expects a 4D BCHW tensor.")

    batch_size, channels, height, width = feature_map.shape
    return feature_map.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)


# Upsample feature maps to a shared grid, then concatenate the patch features.
def concat_patch_features(feature_maps, target="last"):
    if len(feature_maps) == 0:
        raise ValueError("concat_patch_features received an empty feature map list.")

    if target == "last":
        target_h, target_w = feature_maps[-1].shape[-2:]
    elif target == "max":
        target_h = max(fm.shape[-2] for fm in feature_maps)
        target_w = max(fm.shape[-1] for fm in feature_maps)
    elif isinstance(target, (tuple, list)) and len(target) == 2:
        target_h, target_w = int(target[0]), int(target[1])
    else:
        raise ValueError("target must be 'last', 'max', or a (height, width) pair.")

    patch_list = []
    for feature_map in feature_maps:
        if feature_map.ndim != 4:
            raise ValueError("Each feature map must be a 4D BCHW tensor.")

        if feature_map.shape[-2:] != (target_h, target_w):
            feature_map = F.interpolate(
                feature_map,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

        patch_list.append(fmap_to_patches(feature_map))

    return torch.cat(patch_list, dim=-1)


# Return the spatial size of one feature map or of the final map in a list.
def get_feature_grid_size(feature_maps):
    if isinstance(feature_maps, (list, tuple)):
        if len(feature_maps) == 0:
            raise ValueError("Feature map list is empty.")
        feature_map = feature_maps[-1]
    else:
        feature_map = feature_maps

    if feature_map.ndim != 4:
        raise ValueError("get_feature_grid_size expects a 4D BCHW tensor.")

    return int(feature_map.shape[-2]), int(feature_map.shape[-1])


# Return the concatenated patch feature dimension.
def get_patch_feature_dim(feature_maps, target="last"):
    patches = concat_patch_features(feature_maps, target=target)
    return int(patches.shape[-1])


#------------------------------------------------------------------------------
# Checkpoint loading helpers
#------------------------------------------------------------------------------

# Clean a checkpoint state dict so it matches the plain encoder structure.
def clean_state_dict_keys(state_dict):
    clean_state = {}
    for key, value in state_dict.items():
        clean_key = str(key)
        if clean_key.startswith("module."):
            clean_key = clean_key.replace("module.", "", 1)
        if clean_key.startswith("encoder."):
            clean_key = clean_key.replace("encoder.", "", 1)
        clean_state[clean_key] = value
    return clean_state


# Extract the most likely state-dict payload from a saved checkpoint object.
def extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        for key in ["state_dict", "model_state_dict", "encoder_state_dict"]:
            if key in checkpoint_obj and isinstance(checkpoint_obj[key], dict):
                return checkpoint_obj[key]

        tensor_like = [isinstance(v, torch.Tensor) for v in checkpoint_obj.values()]
        if len(tensor_like) > 0 and all(tensor_like):
            return checkpoint_obj

    if isinstance(checkpoint_obj, nn.Module):
        return checkpoint_obj.state_dict()

    raise ValueError(
        "Could not extract a state dict from the supplied checkpoint object. "
        "Expected a raw state dict or a dict containing state_dict/model_state_dict/encoder_state_dict."
    )


# Load a checkpoint into an already-built model while cleaning common prefixes.
def load_model_weights_flexible(model, checkpoint_path, strict=True, map_location="cpu"):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_obj = torch.load(checkpoint_path, map_location=map_location)
    state_dict = extract_state_dict(checkpoint_obj)
    state_dict = clean_state_dict_keys(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    return {
        "model": model,
        "checkpoint_path": str(checkpoint_path),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "n_loaded_tensors": len(state_dict),
    }


# Rebuild one SSL encoder checkpoint into the shared ResNet-18 encoder shape.
def load_ssl_encoder(ckpt_path, strict=True, map_location="cpu"):
    model = get_resnet18_ssl()
    _ = load_model_weights_flexible(
        model=model,
        checkpoint_path=ckpt_path,
        strict=strict,
        map_location=map_location,
    )
    model.eval()
    return model


# Return simple checkpoint metadata for logging.
def checkpoint_metadata(ckpt_path):
    ckpt_path = Path(ckpt_path)
    return {
        "checkpoint_path": str(ckpt_path),
        "exists": bool(ckpt_path.exists()),
        "size_mb": float(ckpt_path.stat().st_size / (1024 ** 2)) if ckpt_path.exists() else None,
    }
