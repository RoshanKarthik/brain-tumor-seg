"""Label <-> region (TC, WT, ET) conversion.

BraTS 2021 labels: 0 = background, 1 = NCR, 2 = ED, 4 = ET (BraTS 2023+ uses 3 for ET,
hence ``et_label``). The model predicts three overlapping regions, in this order:

- TC (tumor core)    = label in {1, ET}
- WT (whole tumor)   = label in {1, 2, ET}
- ET (enhancing)     = label == ET

Works on numpy arrays and torch tensors (including MONAI MetaTensors).
"""

from __future__ import annotations

import numpy as np
import torch

REGIONS = ("TC", "WT", "ET")
NCR_LABEL = 1
ED_LABEL = 2


def _squeeze_channel(x):
    if x.ndim == 4 and x.shape[0] == 1:
        return x[0]
    if x.ndim != 3:
        raise ValueError(
            f"Expected a (H, W, D) or (1, H, W, D) label map, got shape {tuple(x.shape)}"
        )
    return x


def labels_to_regions(seg, et_label: int = 4):
    """Convert a label map to boolean region masks of shape (3, H, W, D): [TC, WT, ET].

    Raises ``ValueError`` if the map contains labels other than {0, 1, 2, et_label}.
    """
    seg = _squeeze_channel(seg)
    allowed = {0, NCR_LABEL, ED_LABEL, et_label}
    if isinstance(seg, torch.Tensor):
        found = {int(v) for v in torch.unique(seg).tolist()}
        stack = torch.stack
    else:
        seg = np.asarray(seg)
        found = {int(v) for v in np.unique(seg)}
        stack = np.stack
    unexpected = found - allowed
    if unexpected:
        raise ValueError(
            f"Unexpected label values {sorted(unexpected)}; allowed {sorted(allowed)} "
            f"(et_label={et_label})"
        )

    et = seg == et_label
    tc = (seg == NCR_LABEL) | et
    wt = tc | (seg == ED_LABEL)
    return stack([tc, wt, et])


def regions_to_labels(regions, et_label: int = 4):
    """Convert binary region masks (3, H, W, D) [TC, WT, ET] to a (H, W, D) label map.

    Uses the nested hierarchy: start from 0; WT -> 2 (ED); TC -> 1 (NCR); ET -> et_label.
    Later steps overwrite earlier ones, so the result always satisfies ET ⊆ TC ⊆ WT even
    if the input masks are not nested. Inputs must already be binary (bool or 0/1);
    thresholding probabilities is the caller's job.
    """
    if regions.ndim != 4 or regions.shape[0] != 3:
        raise ValueError(f"Expected regions of shape (3, H, W, D), got {tuple(regions.shape)}")
    if isinstance(regions, torch.Tensor):
        masks = regions.bool()
        labels = torch.zeros(regions.shape[1:], dtype=torch.uint8, device=regions.device)
    else:
        masks = np.asarray(regions).astype(bool)
        labels = np.zeros(masks.shape[1:], dtype=np.uint8)
    tc, wt, et = masks[0], masks[1], masks[2]
    labels[wt] = ED_LABEL
    labels[tc] = NCR_LABEL
    labels[et] = et_label
    return labels
