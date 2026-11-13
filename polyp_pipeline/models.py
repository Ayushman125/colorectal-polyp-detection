from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ModelSpec:
    architecture: str
    encoder: str


MODEL_SPECS = {
    "unet_resnet34": ModelSpec("Unet", "resnet34"),
    "unet_resnet50": ModelSpec("Unet", "resnet50"),
    "unet_efficientnet_b0": ModelSpec("Unet", "efficientnet-b0"),
    "unetpp_resnet34": ModelSpec("UnetPlusPlus", "resnet34"),
    "unetpp_resnet50": ModelSpec("UnetPlusPlus", "resnet50"),
    "unetpp_efficientnet_b0": ModelSpec("UnetPlusPlus", "efficientnet-b0"),
    "fpn_resnet34": ModelSpec("FPN", "resnet34"),
    "deeplabv3plus_resnet34": ModelSpec("DeepLabV3Plus", "resnet34"),
    "segformer_mit_b0": ModelSpec("Segformer", "mit_b0"),
    "upernet_mit_b0": ModelSpec("UPerNet", "mit_b0"),
}


def build_model(name: str, pretrained_encoder: bool = False) -> nn.Module:
    """Build a real SMP segmentation model with a single foreground logit."""
    if name not in MODEL_SPECS:
        raise ValueError(
            f"Unknown model {name!r}. Available models: {', '.join(MODEL_SPECS)}"
        )
    try:
        import segmentation_models_pytorch as smp
    except ImportError as error:
        raise RuntimeError(
            "Install the dependencies from requirements.txt to use deep models."
        ) from error

    spec = MODEL_SPECS[name]
    constructor = getattr(smp, spec.architecture)
    return constructor(
        encoder_name=spec.encoder,
        encoder_weights="imagenet" if pretrained_encoder else None,
        in_channels=3,
        classes=1,
        activation=None,
    )


def compound_bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    probabilities = torch.sigmoid(logits)
    dimensions = (1, 2, 3)
    intersection = (probabilities * target).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + target.sum(dim=dimensions)
    dice_loss = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    return bce + dice_loss.mean()


def weighted_bce_iou_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weights = 1.0 + 5.0 * torch.abs(
        nn.functional.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target
    )
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    weighted_bce = (weights * bce).sum(dim=(2, 3)) / weights.sum(dim=(2, 3))
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * target * weights).sum(dim=(2, 3))
    union = ((probabilities + target) * weights).sum(dim=(2, 3))
    weighted_iou = 1.0 - (intersection + 1.0) / (union - intersection + 1.0)
    return (weighted_bce + weighted_iou).mean()


def focal_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 0.75,
    smooth: float = 1.0,
) -> torch.Tensor:
    if alpha < 0 or beta < 0 or gamma <= 0 or smooth <= 0:
        raise ValueError("Tversky weights must be non-negative and gamma/smooth positive")
    probabilities = torch.sigmoid(logits)
    dimensions = (1, 2, 3)
    true_positive = (probabilities * target).sum(dim=dimensions)
    false_positive = (probabilities * (1.0 - target)).sum(dim=dimensions)
    false_negative = ((1.0 - probabilities) * target).sum(dim=dimensions)
    tversky = (true_positive + smooth) / (
        true_positive + alpha * false_positive + beta * false_negative + smooth
    )
    return (1.0 - tversky).clamp_min(0.0).pow(gamma).mean()


def dice_boundary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    boundary_weight: float = 0.1,
) -> torch.Tensor:
    if boundary_weight < 0:
        raise ValueError("boundary_weight must be non-negative")
    from scipy.ndimage import distance_transform_edt

    probabilities = torch.sigmoid(logits)
    dimensions = (1, 2, 3)
    intersection = (probabilities * target).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + target.sum(dim=dimensions)
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    binary_targets = target.detach().to(device="cpu").numpy()[:, 0] > 0.5
    signed_distances = []
    normalizer = max(target.shape[-2:])
    for binary_target in binary_targets:
        distance_inside = distance_transform_edt(binary_target)
        distance_outside = distance_transform_edt(~binary_target)
        signed = (distance_outside - distance_inside) / normalizer
        signed_distances.append(signed.astype("float32"))
    distance_map = torch.as_tensor(
        np.stack(signed_distances),
        dtype=probabilities.dtype,
        device=probabilities.device,
    )[:, None]
    boundary = (probabilities * distance_map).mean(dim=dimensions)
    return (1.0 - dice + boundary_weight * boundary).mean()


def segmentation_loss(name: str, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if name == "bce":
        return nn.functional.binary_cross_entropy_with_logits(logits, target)
    if name == "bce_dice":
        return compound_bce_dice_loss(logits, target)
    if name == "weighted_bce_iou":
        return weighted_bce_iou_loss(logits, target)
    if name == "focal_tversky":
        return focal_tversky_loss(logits, target)
    if name == "dice_boundary":
        return dice_boundary_loss(logits, target)
    raise ValueError(f"Unsupported loss: {name}")
