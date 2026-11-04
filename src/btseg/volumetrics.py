"""Voxel volume from the affine and per-region tumor volumes in cm^3 (= mL).

The voxel volume is ``|det(affine[:3, :3])|`` in mm^3, never an assumed 1 mm^3, so
anisotropic or resampled uploads are measured correctly.
"""

from __future__ import annotations

import numpy as np

from btseg.data.labels import ED_LABEL, NCR_LABEL

VOLUME_REGIONS = ("NCR", "ED", "ET", "TC", "WT")


def voxel_volume_mm3(affine) -> float:
    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 affine, got shape {affine.shape}")
    volume = abs(float(np.linalg.det(affine[:3, :3])))
    if volume == 0:
        raise ValueError("Affine has zero voxel volume")
    return volume


def region_voxel_counts(label_map, et_label: int = 4) -> dict[str, int]:
    """Voxel counts for NCR, ED, ET and the composites TC = NCR + ET, WT = NCR + ED + ET."""
    seg = np.asarray(label_map)
    ncr = int(np.count_nonzero(seg == NCR_LABEL))
    ed = int(np.count_nonzero(seg == ED_LABEL))
    et = int(np.count_nonzero(seg == et_label))
    return {"NCR": ncr, "ED": ed, "ET": et, "TC": ncr + et, "WT": ncr + ed + et}


def region_volumes(label_map, affine, et_label: int = 4) -> dict[str, dict[str, float]]:
    """``{region: {"voxels": n, "cm3": v}}`` for NCR, ED, ET, TC and WT."""
    mm3 = voxel_volume_mm3(affine)
    counts = region_voxel_counts(label_map, et_label)
    return {r: {"voxels": counts[r], "cm3": counts[r] * mm3 / 1000.0} for r in VOLUME_REGIONS}
