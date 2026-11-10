import numpy as np
import pytest

from btseg.config import PostprocessConfig
from btseg.data.labels import labels_to_regions, regions_to_labels
from btseg.postprocess import enforce_hierarchy, postprocess, remove_small_components


def blank(shape=(20, 20, 20)):
    return np.zeros((3, *shape), np.float32)


def test_threshold_and_hierarchy():
    probs = blank()
    probs[2, 5:8, 5:8, 5:8] = 0.9  # ET predicted, but TC / WT below threshold there
    probs[1, 2:4, 2:4, 2:4] = 0.6  # WT only
    probs[0, 10, 10, 10] = 0.5  # exactly at threshold: not included (> 0.5)
    tc, wt, et = postprocess(probs)
    assert et.sum() == 27
    assert (tc >= et).all() and (wt >= tc).all()
    assert wt.sum() == 27 + 8
    assert not tc[10, 10, 10]
    labels = regions_to_labels(np.stack([tc, wt, et]))
    np.testing.assert_array_equal(labels_to_regions(labels), np.stack([tc, wt, et]))


def test_remove_small_components():
    mask = np.zeros((20, 20, 20), bool)
    mask[1:6, 1:6, 1:6] = True  # 125 voxels
    mask[15, 15, 15] = True  # 1 voxel
    mask[10, 10, 10] = mask[11, 11, 11] = True  # diagonal neighbours: one 2-voxel component
    out = remove_small_components(mask, 3)
    assert out.sum() == 125 and not out[15, 15, 15] and not out[10, 10, 10]
    assert remove_small_components(mask, 2).sum() == 127
    assert remove_small_components(mask, 0) is mask


def test_small_component_removal_is_off_by_default_and_reenforces_hierarchy():
    probs = blank()
    probs[1, 1:6, 1:6, 1:6] = 0.9  # big WT
    probs[0, 15, 15, 15] = probs[1, 15, 15, 15] = 0.9  # isolated 1-voxel TC (+WT)
    probs[0, 3, 3, 3] = 0.9  # TC voxel inside the big WT
    assert postprocess(probs)[0].sum() == 2
    tc, wt, et = postprocess(probs, PostprocessConfig(min_component_voxels=5))
    assert tc.sum() == 0 and wt.sum() == 125  # the 1-voxel TC is gone from both
    assert not et.any()


def test_small_et_becomes_ncr():
    probs = blank()
    probs[1, 2:12, 2:12, 2:12] = 0.9
    probs[0, 4:8, 4:8, 4:8] = 0.9
    probs[2, 5:7, 5:7, 5:7] = 0.9  # 8 ET voxels
    cfg = PostprocessConfig(et_min_voxels=10)
    tc, wt, et = postprocess(probs, cfg)
    assert et.sum() == 0 and tc.sum() == 64  # ET voxels stayed in TC -> NCR
    assert (regions_to_labels(np.stack([tc, wt, et]))[5:7, 5:7, 5:7] == 1).all()
    assert postprocess(probs, PostprocessConfig(et_min_voxels=8))[2].sum() == 8


def test_bad_input_shape():
    with pytest.raises(ValueError):
        postprocess(np.zeros((2, 4, 4, 4)))


def test_enforce_hierarchy_does_not_remove():
    r = np.zeros((3, 4, 4, 4), bool)
    r[2, 0, 0, 0] = True
    out = enforce_hierarchy(r)
    assert out[:, 0, 0, 0].all()


def test_region_thresholds_override_the_global_threshold():
    probs = blank()
    probs[1, 2:6, 2:6, 2:6] = 0.4  # WT probability below 0.5
    assert postprocess(probs)[1].sum() == 0
    cfg = PostprocessConfig(region_thresholds=(0.5, 0.35, 0.5))
    tc, wt, et = postprocess(probs, cfg)
    assert wt.sum() == 64 and tc.sum() == 0 and et.sum() == 0
