import json

import nibabel as nib
import numpy as np
import pytest
import torch
from torch import nn

from btseg.config import Config
from btseg.data.discover import SEQUENCES
from btseg.infer import FLIP_DIMS, Predictor, main, validate_inputs
from btseg.models.factory import build_model
from conftest import write_case

# flipped x/y (LPS-like) and anisotropic voxels: exercises orientation + spacing inverses
LPS_ANISO = np.array(
    [[-1.5, 0, 0, 60.0], [0, -1.5, 0, 50.0], [0, 0, 2.5, -20.0], [0, 0, 0, 1]]
)


class ThresholdModel(nn.Module):
    """Predicts every region where the normalized FLAIR is > 1 (the bright tumor)."""

    def forward(self, x):
        return (20.0 * (x[:, :1] - 1.0)).repeat(1, 3, 1, 1, 1)


def dice(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return 2 * (a & b).sum() / (a.sum() + b.sum())


def files(case):
    return [case[s] for s in SEQUENCES]


def test_prediction_maps_back_to_the_input_space(tmp_path):
    case = write_case(tmp_path, "c", shape=(48, 44, 28), affine=LPS_ANISO)
    result = Predictor(ThresholdModel(), Config(), device="cpu").predict(*files(case))

    seg = nib.load(case["seg"]).get_fdata()
    assert result.label_map.shape == seg.shape and result.label_map.dtype == np.uint8
    assert result.probabilities.shape == (3, *seg.shape)
    np.testing.assert_allclose(result.affine, LPS_ANISO)
    assert 0.0 <= result.probabilities.min() and result.probabilities.max() <= 1.0
    # the bright tumor is found at the same voxels (orientation and spacing were undone)
    assert dice(result.label_map > 0, seg > 0) > 0.9
    assert result.uncertainty is None and not result.tta
    assert result.volumes["WT"]["cm3"] == pytest.approx(
        (result.label_map > 0).sum() * 1.5 * 1.5 * 2.5 / 1000
    )


def test_tta_averages_8_flips_and_reports_variance(tmp_path):
    assert len(FLIP_DIMS) == 8 and len(set(FLIP_DIMS)) == 8
    case = write_case(tmp_path, "c", shape=(32, 32, 32))
    torch.manual_seed(0)
    model = build_model("segresnet", init_filters=8)
    result = Predictor(model, Config(), device="cpu").predict(*files(case), tta=True)
    assert result.tta
    assert result.uncertainty.shape == result.probabilities.shape == (3, 32, 32, 32)
    assert result.uncertainty.min() >= 0 and result.uncertainty.max() <= 0.25
    assert result.uncertainty.max() > 0  # a random model is not flip-invariant


def test_from_checkpoint_and_cli(tmp_path):
    torch.manual_seed(0)
    cfg = Config()
    cfg.model.kwargs = {"init_filters": 8}
    model = build_model("segresnet", init_filters=8)
    ckpt = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "epoch": 1}, ckpt)

    case = write_case(tmp_path / "case", "BraTS2021_99999", shape=(32, 32, 24), with_seg=False)
    report = main(["--ckpt", str(ckpt), "--case-dir", str(tmp_path / "case"),
                   "--out", str(tmp_path / "out"), "--device", "cpu"])

    seg = nib.load(tmp_path / "out" / "seg.nii.gz")
    flair = nib.load(case["flair"])
    assert seg.shape == flair.shape
    np.testing.assert_allclose(seg.affine, flair.affine)
    assert seg.get_data_dtype() == np.uint8
    assert set(np.unique(np.asarray(seg.dataobj))) <= {0, 1, 2, 4}
    saved = json.loads((tmp_path / "out" / "volumes.json").read_text())
    assert saved == json.loads(json.dumps(report))
    assert set(saved["volumes"]) == {"NCR", "ED", "ET", "TC", "WT"}
    assert saved["case_id"] == "BraTS2021_99999" and saved["model"] == "segresnet"


def test_validate_inputs_rejects_mismatched_files(tmp_path):
    a = write_case(tmp_path / "a", "a", shape=(16, 16, 16), with_seg=False)
    b = write_case(tmp_path / "b", "b", shape=(16, 16, 20), with_seg=False)
    c = write_case(tmp_path / "c", "c", shape=(16, 16, 16), spacing=(1, 1, 2), with_seg=False)
    validate_inputs(files(a))
    with pytest.raises(ValueError, match="shape"):
        validate_inputs([*files(a)[:3], b["t2"]])
    with pytest.raises(ValueError, match="affine"):
        validate_inputs([*files(a)[:3], c["t2"]])
    with pytest.raises(ValueError, match="Expected 4"):
        validate_inputs(files(a)[:3])
    bad = tmp_path / "x_t2.nii.gz"
    bad.write_bytes(b"not a nifti")
    with pytest.raises(ValueError, match="cannot read"):
        validate_inputs([*files(a)[:3], bad])


def test_tta_pass_also_returns_the_plain_prediction(tmp_path):
    case = write_case(tmp_path, "c", shape=(32, 32, 32))
    torch.manual_seed(0)
    predictor = Predictor(build_model("segresnet", init_filters=8), Config(), device="cpu")
    plain = predictor.predict_maps(files(case)).plain
    both = predictor.predict_maps(files(case), tta=True)
    np.testing.assert_allclose(both.plain, plain, atol=1e-5)
    assert not np.allclose(both.tta_mean, plain)
