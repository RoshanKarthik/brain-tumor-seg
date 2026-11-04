import math

import numpy as np
import pytest

from btseg.volumetrics import region_volumes, region_voxel_counts, voxel_volume_mm3


def sphere_mask(shape, spacing, radius_mm):
    """Voxels whose centre lies within ``radius_mm`` of the volume centre (in mm)."""
    axes = [(np.arange(n) - (n - 1) / 2) * s for n, s in zip(shape, spacing, strict=True)]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    return x**2 + y**2 + z**2 <= radius_mm**2


@pytest.mark.parametrize(
    "spacing, shape",
    [((1.0, 1.0, 1.0), (64, 64, 64)), ((0.5, 0.5, 2.0), (128, 128, 32))],
)
def test_sphere_volume_within_2_percent(spacing, shape):
    radius = 20.0
    seg = sphere_mask(shape, spacing, radius).astype(np.uint8) * 2  # all edema
    affine = np.diag([*spacing, 1.0])
    vols = region_volumes(seg, affine)
    expected_cm3 = 4 / 3 * math.pi * radius**3 / 1000
    assert vols["WT"]["cm3"] == pytest.approx(expected_cm3, rel=0.02)
    assert vols["ED"]["cm3"] == vols["WT"]["cm3"]
    assert vols["TC"]["voxels"] == vols["ET"]["voxels"] == vols["NCR"]["voxels"] == 0


def test_voxel_volume_uses_determinant_not_spacing_guess():
    assert voxel_volume_mm3(np.diag([0.5, 0.5, 2.0, 1.0])) == pytest.approx(0.5)
    # flipped axes (e.g. LPS) give a negative determinant; volume stays positive
    assert voxel_volume_mm3(np.diag([-1.0, -1.0, 1.0, 1.0])) == pytest.approx(1.0)
    rot = np.eye(4)
    rot[:2, :2] = [[0.0, -1.2], [1.2, 0.0]]  # 90 degree rotation, 1.2 mm in-plane
    assert voxel_volume_mm3(rot) == pytest.approx(1.44)
    with pytest.raises(ValueError):
        voxel_volume_mm3(np.zeros((4, 4)))


@pytest.mark.parametrize("et_label", [4, 3])
def test_region_counts_and_composites(et_label):
    seg = np.zeros((10, 10, 10), np.uint8)
    seg[0, :3] = 1  # 30 NCR
    seg[1, :5] = 2  # 50 ED
    seg[2, :2] = et_label  # 20 ET
    assert region_voxel_counts(seg, et_label) == {
        "NCR": 30, "ED": 50, "ET": 20, "TC": 50, "WT": 100,
    }
    vols = region_volumes(seg, np.diag([2.0, 2.0, 2.0, 1.0]), et_label)
    assert vols["WT"] == {"voxels": 100, "cm3": pytest.approx(0.8)}  # 100 * 8 mm^3
