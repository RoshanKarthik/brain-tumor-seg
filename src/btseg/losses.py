"""Loss selection: DiceLoss or DiceFocalLoss on sigmoid outputs (multi-label TC/WT/ET).

Targets are the float (B, 3, H, W, D) region masks from ``LabelsToRegionsd``; inputs are
raw logits. ``kwargs`` from the config override the defaults below.

Empty regions: many random 128^3 crops contain no ET (some cases have none at all).
With MONAI's defaults (``smooth_nr = smooth_dr = 1e-5``, linear denominator) a correct
"nothing here" prediction still scores a loss of ~1, because the residual sigmoid mass of
~2M background voxels (e.g. 2M x sigmoid(-10) ~ 95 voxels) dominates the denominator.
So the defaults here are ``squared_pred=True`` (V-Net form; residual mass becomes
2M x sigmoid(-10)^2 ~ 0.004) and ``smooth_nr = smooth_dr = 1``, which give ~0 loss for a
correctly empty region, ~1 for a false-positive blob, and are negligible for real tumor
regions (thousands of voxels). All of them can be overridden from the config.
"""

from __future__ import annotations

from typing import Any

from monai.losses import DiceFocalLoss, DiceLoss
from torch import nn

_LOSSES = {"dice": DiceLoss, "dicefocal": DiceFocalLoss}
_DEFAULTS = {"sigmoid": True, "squared_pred": True, "smooth_nr": 1.0, "smooth_dr": 1.0}


def build_loss(name: str, **kwargs: Any) -> nn.Module:
    if name not in _LOSSES:
        raise ValueError(f"Unknown loss {name!r}; expected one of {sorted(_LOSSES)}")
    params = {**_DEFAULTS, **kwargs}
    if not params["sigmoid"] or params.get("softmax"):
        raise ValueError("Regions overlap (ET ⊆ TC ⊆ WT), so the loss must use sigmoid outputs")
    return _LOSSES[name](**params)
