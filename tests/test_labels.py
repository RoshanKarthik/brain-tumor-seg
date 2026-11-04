import numpy as np
import pytest
import torch

from btseg.data.labels import labels_to_regions, regions_to_labels


def random_labels(shape=(16, 16, 16), et_label=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([0, 1, 2, et_label], dtype=np.uint8), size=shape)


def assert_nested(regions):
    tc, wt, et = (np.asarray(r, dtype=bool) for r in regions)
    assert not (et & ~tc).any(), "ET must be inside TC"
    assert not (tc & ~wt).any(), "TC must be inside WT"


@pytest.mark.parametrize("et_label", [4, 3])
def test_round_trip_numpy(et_label):
    seg = random_labels(et_label=et_label)
    regions = labels_to_regions(seg, et_label=et_label)
    assert regions.shape == (3, *seg.shape)
    assert regions.dtype == bool
    np.testing.assert_array_equal(regions_to_labels(regions, et_label=et_label), seg)


def test_round_trip_torch_with_channel_dim():
    seg = torch.from_numpy(random_labels(seed=1)).unsqueeze(0).float()
    regions = labels_to_regions(seg)
    assert isinstance(regions, torch.Tensor)
    assert regions.shape == (3, 16, 16, 16)
    back = regions_to_labels(regions)
    assert torch.equal(back, seg[0].to(torch.uint8))


def test_region_definitions():
    seg = np.array([0, 1, 2, 4], dtype=np.uint8).reshape(1, 1, 4)
    tc, wt, et = labels_to_regions(seg)
    np.testing.assert_array_equal(tc.ravel(), [0, 1, 0, 1])
    np.testing.assert_array_equal(wt.ravel(), [0, 1, 1, 1])
    np.testing.assert_array_equal(et.ravel(), [0, 0, 0, 1])


def test_labels_to_regions_is_nested():
    assert_nested(labels_to_regions(random_labels(seed=2)))


def test_regions_to_labels_enforces_hierarchy_on_non_nested_masks():
    rng = np.random.default_rng(3)
    regions = rng.random((3, 16, 16, 16)) > 0.5  # arbitrary, not nested
    labels = regions_to_labels(regions)
    assert set(np.unique(labels)) <= {0, 1, 2, 4}
    assert_nested(labels_to_regions(labels))
    # ET wins, then TC, then WT
    np.testing.assert_array_equal(labels == 4, regions[2])
    np.testing.assert_array_equal(labels == 1, regions[0] & ~regions[2])


def test_unexpected_label_raises():
    seg = np.array([0, 1, 2, 3], dtype=np.uint8).reshape(1, 1, 4)
    with pytest.raises(ValueError, match="Unexpected label"):
        labels_to_regions(seg, et_label=4)


def test_bad_shapes_raise():
    with pytest.raises(ValueError):
        labels_to_regions(np.zeros((2, 4, 4, 4)))
    with pytest.raises(ValueError):
        regions_to_labels(np.zeros((2, 4, 4, 4)))
