"""Shared image transforms for the MVTec AD project.

This module keeps together the transform logic used across the notebooks.
It focuses on:
- standard evaluation transforms for ImageNet-based feature methods
- simple 0-1 transforms for the autoencoder baseline
- SimCLR mild and strong augmentation policies
- small display helpers for undoing ImageNet normalisation

The goal is to keep transform choices consistent across baselines, SSL, and
qualitative plotting.
"""

import random

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.transforms import InterpolationMode


#------------------------------------------------------------------------------
# Shared constants
#------------------------------------------------------------------------------

# Standard ImageNet statistics used by the feature-based methods.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Default interpolation choices kept explicit so resizing behaviour is stable.
RGB_INTERPOLATION = InterpolationMode.BILINEAR
MASK_INTERPOLATION = InterpolationMode.NEAREST


#------------------------------------------------------------------------------
# Small helper classes
#------------------------------------------------------------------------------

# Apply PIL Gaussian blur with a probability. This is used in the SimCLR stage.
class GaussianBlur:
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.p = float(p)
        self.radius_min = float(radius_min)
        self.radius_max = float(radius_max)

    def __call__(self, img):
        if random.random() > self.p:
            return img
        radius = float(random.uniform(self.radius_min, self.radius_max))
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    def __repr__(self):
        return (
            f"GaussianBlur(p={self.p}, radius_min={self.radius_min}, "
            f"radius_max={self.radius_max})"
        )


# Return two independently augmented views from one base image.
class TwoViewTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, img):
        x1 = self.base_transform(img)
        x2 = self.base_transform(img)
        return x1, x2

    def __repr__(self):
        return f"TwoViewTransform(base_transform={self.base_transform})"


#------------------------------------------------------------------------------
# Basic transform helpers
#------------------------------------------------------------------------------

# Return the shared ImageNet statistics as simple Python lists.
def get_imagenet_stats():
    return list(IMAGENET_MEAN), list(IMAGENET_STD)


# Build the ImageNet normalisation layer used by PatchCore / PaDiM / SimCLR.
def get_imagenet_normalize(mean=None, std=None):
    mean = IMAGENET_MEAN if mean is None else list(mean)
    std = IMAGENET_STD if std is None else list(std)
    return transforms.Normalize(mean, std)


# Build the resize+tensor transform used by the ImageNet feature methods.
def build_imagenet_eval_transform(img_size, mean=None, std=None):
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=RGB_INTERPOLATION),
        transforms.ToTensor(),
        get_imagenet_normalize(mean=mean, std=std),
    ])


# Build the resize+tensor transform used by the autoencoder baseline.
def build_autoencoder_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=RGB_INTERPOLATION),
        transforms.ToTensor(),
    ])


# Small wrapper so notebooks can request transforms by method family.
def build_eval_transform(img_size, mode="imagenet", mean=None, std=None):
    mode = str(mode).strip().lower()

    if mode in ["imagenet", "feature", "feature_based", "patchcore", "padim"]:
        return build_imagenet_eval_transform(img_size, mean=mean, std=std)
    if mode in ["autoencoder", "ae", "raw"]:
        return build_autoencoder_transform(img_size)

    raise ValueError(f"Unknown eval transform mode: {mode}")


#------------------------------------------------------------------------------
# SimCLR augmentation helpers
#------------------------------------------------------------------------------

# Return the augmentation settings used in the project for one SSL strength.
def get_simclr_policy(strength="mild"):
    strength = str(strength).strip().lower()

    if strength == "mild":
        return {
            "strength": "mild",
            "color_jitter": (0.20, 0.20, 0.10, 0.05),
            "blur_p": 0.20,
            "blur_radius_min": 0.1,
            "blur_radius_max": 1.0,
            "crop_scale": (0.60, 1.00),
            "grayscale_p": 0.05,
            "flip_p": 0.50,
            "random_apply_p": 0.80,
        }

    if strength == "strong":
        return {
            "strength": "strong",
            "color_jitter": (0.40, 0.40, 0.20, 0.10),
            "blur_p": 0.50,
            "blur_radius_min": 0.1,
            "blur_radius_max": 2.0,
            "crop_scale": (0.40, 1.00),
            "grayscale_p": 0.10,
            "flip_p": 0.50,
            "random_apply_p": 0.80,
        }

    raise ValueError(f"Unknown SimCLR strength: {strength}")


# Build the main SimCLR transform used in the study.
def build_simclr_transform(img_size, strength="mild", mean=None, std=None):
    policy = get_simclr_policy(strength)
    cj = transforms.ColorJitter(*policy["color_jitter"])
    blur = GaussianBlur(
        p=policy["blur_p"],
        radius_min=policy["blur_radius_min"],
        radius_max=policy["blur_radius_max"],
    )

    return transforms.Compose([
        transforms.RandomResizedCrop(
            img_size,
            scale=policy["crop_scale"],
            interpolation=RGB_INTERPOLATION,
        ),
        transforms.RandomHorizontalFlip(p=policy["flip_p"]),
        transforms.RandomApply([cj], p=policy["random_apply_p"]),
        transforms.RandomGrayscale(p=policy["grayscale_p"]),
        blur,
        transforms.ToTensor(),
        get_imagenet_normalize(mean=mean, std=std),
    ])


# Build a two-view wrapper directly for SimCLR datasets.
def build_simclr_two_view_transform(img_size, strength="mild", mean=None, std=None):
    base_transform = build_simclr_transform(
        img_size=img_size,
        strength=strength,
        mean=mean,
        std=std,
    )
    return TwoViewTransform(base_transform)


# Return the SimCLR policies in a small table-friendly format.
def simclr_policy_rows(strengths=("mild", "strong")):
    rows = []
    for strength in strengths:
        policy = get_simclr_policy(strength)
        rows.append({
            "strength": policy["strength"],
            "crop_scale_min": float(policy["crop_scale"][0]),
            "crop_scale_max": float(policy["crop_scale"][1]),
            "flip_p": float(policy["flip_p"]),
            "grayscale_p": float(policy["grayscale_p"]),
            "random_apply_p": float(policy["random_apply_p"]),
            "blur_p": float(policy["blur_p"]),
            "blur_radius_min": float(policy["blur_radius_min"]),
            "blur_radius_max": float(policy["blur_radius_max"]),
            "brightness": float(policy["color_jitter"][0]),
            "contrast": float(policy["color_jitter"][1]),
            "saturation": float(policy["color_jitter"][2]),
            "hue": float(policy["color_jitter"][3]),
        })
    return rows


#------------------------------------------------------------------------------
# Display helpers
#------------------------------------------------------------------------------

# Undo ImageNet normalisation on a tensor so it can be plotted more easily.
def denormalize_tensor(x, mean=None, std=None, clamp=True):
    if not torch.is_tensor(x):
        raise TypeError("denormalize_tensor expects a torch.Tensor input.")

    mean = IMAGENET_MEAN if mean is None else list(mean)
    std = IMAGENET_STD if std is None else list(std)

    if x.ndim == 3:
        mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(-1, 1, 1)
        std_t = torch.tensor(std, dtype=x.dtype, device=x.device).view(-1, 1, 1)
    elif x.ndim == 4:
        mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        std_t = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    else:
        raise ValueError("denormalize_tensor expects a 3D or 4D tensor.")

    out = x * std_t + mean_t
    if clamp:
        out = out.clamp(0.0, 1.0)
    return out


# Convert a CHW or BCHW tensor into a NumPy array for plotting.
def tensor_to_numpy_image(x, mean=None, std=None, denormalize=False, index=0):
    if not torch.is_tensor(x):
        raise TypeError("tensor_to_numpy_image expects a torch.Tensor input.")

    if x.ndim == 4:
        x = x[index]

    if x.ndim != 3:
        raise ValueError("tensor_to_numpy_image expects a CHW or BCHW tensor.")

    if denormalize:
        x = denormalize_tensor(x, mean=mean, std=std, clamp=True)

    arr = x.detach().cpu().permute(1, 2, 0).numpy()
    arr = np.clip(arr, 0.0, 1.0)
    return arr


# Convert a single-channel mask tensor to a NumPy array for plotting.
def mask_tensor_to_numpy(mask_tensor, index=0):
    if not torch.is_tensor(mask_tensor):
        raise TypeError("mask_tensor_to_numpy expects a torch.Tensor input.")

    if mask_tensor.ndim == 4:
        mask_tensor = mask_tensor[index]

    if mask_tensor.ndim == 3 and mask_tensor.shape[0] == 1:
        mask_tensor = mask_tensor[0]

    if mask_tensor.ndim != 2:
        raise ValueError("mask_tensor_to_numpy expects a HW, 1HW, or B1HW tensor.")

    return mask_tensor.detach().cpu().numpy()


#------------------------------------------------------------------------------
# Small convenience helpers
#------------------------------------------------------------------------------

# Return a compact transform bundle often used by the notebooks.
def build_project_transforms(img_size):
    return {
        "imagenet_eval": build_imagenet_eval_transform(img_size),
        "autoencoder": build_autoencoder_transform(img_size),
        "simclr_mild": build_simclr_transform(img_size, strength="mild"),
        "simclr_strong": build_simclr_transform(img_size, strength="strong"),
    }


# Return a lightweight configuration dictionary for logging.
def transform_config_dict(img_size):
    return {
        "img_size": int(img_size),
        "imagenet_mean": list(IMAGENET_MEAN),
        "imagenet_std": list(IMAGENET_STD),
        "simclr_mild": get_simclr_policy("mild"),
        "simclr_strong": get_simclr_policy("strong"),
    }
