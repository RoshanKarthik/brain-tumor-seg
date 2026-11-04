"""Turn region probabilities [TC, WT, ET] into binary, consistent region masks.

Steps (each controlled by :class:`btseg.config.PostprocessConfig`, so they can be ablated):

1. threshold the probabilities (default 0.5);
2. enforce the hierarchy ET ⊆ TC ⊆ WT by propagating upwards (TC |= ET, WT |= TC),
   the same rule :func:`btseg.data.labels.regions_to_labels` applies;
3. optionally remove connected components smaller than ``min_component_voxels`` from
   each region (26-connectivity), then re-enforce the hierarchy;
4. optionally, if the whole predicted ET is smaller than ``et_min_voxels``, relabel it as
   NCR (it stays in TC). Tiny ET predictions are usually false positives, and an empty
   ET counts as Dice 1 when the ground truth has no ET either.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from btseg.config import PostprocessConfig

_STRUCTURE = np.ones((3, 3, 3), dtype=bool)


def enforce_hierarchy(regions: np.ndarray) -> np.ndarray:
    tc, wt, et = regions
    tc = tc | et
    wt = wt | tc
    return np.stack([tc, wt, et])


def remove_small_components(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    if min_voxels <= 0 or not mask.any():
        return mask
    labeled, n = ndimage.label(mask, structure=_STRUCTURE)
    sizes = np.bincount(labeled.ravel(), minlength=n + 1)
    keep = sizes >= min_voxels
    keep[0] = False
    return keep[labeled]


def postprocess(probabilities, cfg: PostprocessConfig | None = None) -> np.ndarray:
    """(3, H, W, D) probabilities -> (3, H, W, D) bool regions [TC, WT, ET]."""
    cfg = cfg or PostprocessConfig()
    probs = np.asarray(probabilities)
    if probs.ndim != 4 or probs.shape[0] != 3:
        raise ValueError(f"Expected (3, H, W, D) probabilities, got {probs.shape}")
    regions = enforce_hierarchy(probs > cfg.threshold)
    if cfg.min_component_voxels > 0:
        regions = np.stack([remove_small_components(r, cfg.min_component_voxels) for r in regions])
        regions = enforce_hierarchy(regions)
    if cfg.et_min_voxels > 0 and 0 < regions[2].sum() < cfg.et_min_voxels:
        regions[2] = False  # those voxels remain in TC (and WT), i.e. become NCR
    return regions
