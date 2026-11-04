import numpy as np
import pytest

from btseg.metrics import (
    BRATS_HD95_ONE_EMPTY,
    dice,
    hd95,
    region_metrics,
    voxel_spacing,
)


def cube(lo, hi, shape=(20, 20, 20), axis_shift=0):
    m = np.zeros(shape, bool)
    m[lo + axis_shift : hi + axis_shift, lo:hi, lo:hi] = True
    return m


def test_dice_values_and_empty_conventions():
    a = cube(4, 12)
    assert dice(a, a) == 1.0
    b = np.zeros_like(a)
    b[4:12, 4:12, 4:8] = True  # half of a
    assert dice(a, b) == pytest.approx(2 * 256 / (512 + 256))
    empty = np.zeros_like(a)
    assert dice(empty, empty) == 1.0
    assert dice(a, empty) == dice(empty, a) == 0.0


def test_hd95_values_spacing_and_empty_conventions():
    a, b = cube(4, 12), cube(4, 12, axis_shift=3)
    assert hd95(a, a) == 0.0
    assert hd95(a, b) == pytest.approx(3.0)
    assert hd95(a, b, spacing=(2.0, 1.0, 1.0)) == pytest.approx(6.0)
    empty = np.zeros_like(a)
    assert hd95(empty, empty) == 0.0
    assert hd95(a, empty) == hd95(empty, a) == BRATS_HD95_ONE_EMPTY == 373.13


def test_voxel_spacing_from_affine():
    assert voxel_spacing(np.diag([-1.0, -1.0, 1.0, 1.0])) == (1.0, 1.0, 1.0)
    assert voxel_spacing(np.diag([0.5, 0.5, 2.0, 1.0])) == (0.5, 0.5, 2.0)
    rot = np.eye(4)
    rot[:2, :2] = [[0.0, -1.2], [1.2, 0.0]]
    assert voxel_spacing(rot) == pytest.approx((1.2, 1.2, 1.0))


def test_region_metrics_with_absent_et():
    gt = np.zeros((20, 20, 20), np.uint8)
    gt[4:12, 4:12, 4:12] = 2
    gt[6:10, 6:10, 6:10] = 1  # no ET in the ground truth
    pred = gt.copy()
    m = region_metrics(pred, gt, np.eye(4))
    assert list(m) == ["WT", "TC", "ET"]
    assert m["WT"] == m["TC"] == {"dice": 1.0, "hd95": 0.0}
    assert m["ET"] == {"dice": 1.0, "hd95": 0.0}  # both empty
    pred[7, 7, 7] = 4  # a single false-positive ET voxel
    m = region_metrics(pred, gt, np.eye(4))
    assert m["ET"] == {"dice": 0.0, "hd95": 373.13}
    assert m["TC"]["dice"] == 1.0  # the voxel is still tumor core
