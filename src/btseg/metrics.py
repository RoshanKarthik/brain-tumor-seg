"""Dice and HD95 per region with the BraTS conventions for empty masks.

Regions are evaluated in the order WT, TC, ET on label maps in the original image space.

Edge cases (BraTS convention):
- ground truth and prediction both empty: Dice = 1, HD95 = 0 mm
- exactly one of them empty: Dice = 0, HD95 = 373.13 mm
  (373.13 mm is the diagonal of the 240 x 240 x 155 mm BraTS volume)

HD95 is MONAI's symmetric 95th-percentile Hausdorff distance between the mask surfaces:
the larger of the two directed 95th percentiles, in mm using the voxel spacing from the
affine. (Some tools instead take the 95th percentile of both directions pooled, which can
give slightly smaller values.)
"""

from __future__ import annotations

import numpy as np
import torch
from monai.metrics import compute_hausdorff_distance

from btseg.data.labels import labels_to_regions

EVAL_REGIONS = ("WT", "TC", "ET")
BRATS_HD95_ONE_EMPTY = 373.13


def voxel_spacing(affine) -> tuple[float, float, float]:
    """Voxel size along each array axis (column norms of the affine), in mm."""
    a = np.asarray(affine, dtype=np.float64)[:3, :3]
    return tuple(float(v) for v in np.sqrt((a**2).sum(axis=0)))


def dice(pred, gt) -> float:
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    total = pred.sum() + gt.sum()
    if total == 0:
        return 1.0
    return float(2.0 * np.logical_and(pred, gt).sum() / total)


def hd95(pred, gt, spacing=(1.0, 1.0, 1.0)) -> float:
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    p_empty, g_empty = not pred.any(), not gt.any()
    if p_empty and g_empty:
        return 0.0
    if p_empty or g_empty:
        return BRATS_HD95_ONE_EMPTY
    value = compute_hausdorff_distance(
        torch.from_numpy(pred)[None, None],
        torch.from_numpy(gt)[None, None],
        include_background=True,
        percentile=95,
        spacing=list(spacing),
    )
    return float(value.item())


def region_metrics(pred_labels, gt_labels, affine, et_label: int = 4) -> dict[str, dict]:
    """``{region: {"dice": d, "hd95": h}}`` for WT, TC and ET."""
    spacing = voxel_spacing(affine)
    pred = dict(zip(("TC", "WT", "ET"), labels_to_regions(pred_labels, et_label), strict=True))
    gt = dict(zip(("TC", "WT", "ET"), labels_to_regions(gt_labels, et_label), strict=True))
    return {
        r: {"dice": dice(pred[r], gt[r]), "hd95": hd95(pred[r], gt[r], spacing)}
        for r in EVAL_REGIONS
    }
