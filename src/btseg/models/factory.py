"""build_model(name, **kwargs): segresnet, unet3d, swinunetr.

All models take 4 input channels [FLAIR, T1, T1ce, T2] and output 3 logit channels
[TC, WT, ET] (apply a sigmoid per channel). ``kwargs`` override the defaults below.
"""

from __future__ import annotations

from typing import Any

from monai.networks.nets import SegResNet, SwinUNETR, UNet
from torch import nn

_COMMON = {"spatial_dims": 3, "in_channels": 4, "out_channels": 3}

MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    # main model: light enough for ZeroGPU and the CPU fallback
    "segresnet": {
        "init_filters": 16,
        "blocks_down": (1, 2, 2, 4),
        "blocks_up": (1, 1, 1),
        "dropout_prob": 0.2,
    },
    # baseline, chosen to have a parameter count close to the SegResNet above
    "unet3d": {
        "channels": (16, 32, 64, 128, 256),
        "strides": (2, 2, 2, 2),
        "num_res_units": 2,
    },
    # optional; needs ~15x the parameters of SegResNet
    "swinunetr": {"feature_size": 48},
}

_CLASSES = {"segresnet": SegResNet, "unet3d": UNet, "swinunetr": SwinUNETR}

# Spatial input sizes (patch / ROI) must be divisible by this, per model.
SIZE_DIVISOR = {"segresnet": 8, "unet3d": 16, "swinunetr": 32}


def build_model(name: str, **kwargs: Any) -> nn.Module:
    if name not in _CLASSES:
        raise ValueError(f"Unknown model {name!r}; expected one of {sorted(_CLASSES)}")
    params = {**_COMMON, **MODEL_DEFAULTS[name]}
    # YAML gives lists; MONAI expects tuples for block / channel specs
    params.update({k: tuple(v) if isinstance(v, list) else v for k, v in kwargs.items()})
    return _CLASSES[name](**params)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def check_roi_size(name: str, roi_size) -> None:
    divisor = SIZE_DIVISOR[name]
    if any(int(s) % divisor for s in roi_size):
        raise ValueError(f"{name} needs ROI sizes divisible by {divisor}, got {tuple(roi_size)}")
