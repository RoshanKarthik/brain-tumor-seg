import nibabel as nib
import numpy as np
import pytest
import torch

from btseg.data.dataset import case_to_item
from btseg.data.discover import SEQUENCES
from btseg.data.transforms import (
    get_infer_transforms,
    get_train_transforms,
    get_val_transforms,
)
from conftest import write_case


def assert_nested(regions):
    tc, wt, et = regions.bool()
    assert not (et & ~tc).any()
    assert not (tc & ~wt).any()


def test_train_shapes_and_targets(synthetic_case):
    t = get_train_transforms(roi_size=(32, 32, 32))
    t.set_random_state(seed=0)
    out = t(case_to_item(synthetic_case))
    assert out["image"].shape == (4, 32, 32, 32)
    assert out["label"].shape == (3, 32, 32, 32)
    assert out["image"].dtype == torch.float32
    assert set(torch.unique(out["label"]).tolist()) <= {0.0, 1.0}
    assert_nested(out["label"])


def test_train_pads_when_roi_larger_than_foreground(synthetic_case):
    out = get_train_transforms(roi_size=(96, 96, 96))(case_to_item(synthetic_case))
    assert out["image"].shape == (4, 96, 96, 96)
    assert out["label"].shape == (3, 96, 96, 96)


def test_val_keeps_whole_volume_and_normalizes(synthetic_case):
    out = get_val_transforms()(case_to_item(synthetic_case))
    image, label = out["image"], out["label"]
    assert image.shape == (4, 64, 64, 64)
    assert label.shape == (3, 64, 64, 64)
    assert_nested(label)
    # region voxel counts match the source labels (1 mm RAS input -> no resampling)
    seg = nib.load(synthetic_case["seg"]).get_fdata()
    assert label[1].sum().item() == (seg > 0).sum()
    assert label[2].sum().item() == (seg == 4).sum()
    for c in range(4):
        brain = image[c][image[c] != 0]
        assert abs(brain.mean().item()) < 1e-3
        assert abs(brain.std().item() - 1) < 1e-2


def test_infer_resamples_anisotropic_input_to_1mm(tmp_path):
    shape, spacing = (32, 32, 16), (2.0, 2.0, 4.0)
    case = write_case(tmp_path, "aniso", shape=shape, spacing=spacing, with_seg=False)
    out = get_infer_transforms()(case_to_item(case))
    # MONAI keeps the first/last voxel centres: (n - 1) * spacing / 1 mm + 1 voxels
    expected = tuple(int((n - 1) * s) + 1 for n, s in zip(shape, spacing, strict=True))
    assert out["image"].shape == (4, *expected)
    np.testing.assert_allclose(out["image"].pixdim.numpy(), [1.0, 1.0, 1.0], atol=1e-4)


def test_channel_order_is_flair_t1_t1ce_t2(tmp_path):
    # each sequence gets a zero plane at a different z; z-scoring on nonzero keeps zeros at 0
    shape, affine = (16, 16, 16), np.eye(4)
    item = {"id": "order", "image": []}
    for k, seq in enumerate(SEQUENCES):
        img = np.full(shape, 50.0 + k, dtype=np.float32) + np.arange(16, dtype=np.float32)
        img[:, :, 2 + 3 * k] = 0
        path = tmp_path / f"order_{seq}.nii.gz"
        nib.save(nib.Nifti1Image(img, affine), path)
        item["image"].append(str(path))
    image = get_infer_transforms()(item)["image"]
    for k in range(4):
        zero_planes = [z for z in range(16) if (image[k][:, :, z] == 0).all()]
        assert zero_planes == [2 + 3 * k]


@pytest.mark.parametrize("et_label", [3])
def test_et_label_is_configurable(tmp_path, et_label):
    case = write_case(tmp_path, "brats23", et_label=et_label)
    out = get_val_transforms(et_label=et_label)(case_to_item(case))
    seg = nib.load(case["seg"]).get_fdata()
    assert out["label"][2].sum().item() == (seg == et_label).sum() > 0


def test_reorients_to_ras(tmp_path):
    shape = (8, 10, 12)
    lps = np.diag([-1.0, -1.0, 1.0, 1.0])  # x and y axes point left / posterior
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1
    item = {"id": "lps", "image": []}
    for seq in SEQUENCES:
        path = tmp_path / f"lps_{seq}.nii.gz"
        nib.save(nib.Nifti1Image(data, lps), path)
        item["image"].append(str(path))
    image = get_infer_transforms()(item)["image"]
    assert image.shape == (4, *shape)
    assert (np.diag(image.affine.numpy())[:3] > 0).all()
    # flipping x and y back must recover the original voxel order (up to z-scoring)
    z = (data - data.mean()) / data.std()
    np.testing.assert_allclose(image[0].numpy()[::-1, ::-1, :], z, atol=1e-4)
