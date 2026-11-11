"""App smoke test: the Gradio callbacks run end to end on CPU with a tiny random model."""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from btseg.config import Config
from btseg.data.discover import SEQUENCES
from btseg.models.factory import build_model
from conftest import write_case

ROOT = Path(__file__).resolve().parents[1]


def load_app(monkeypatch, checkpoint=None):
    if checkpoint:
        monkeypatch.setenv("BTSEG_CHECKPOINT", str(checkpoint))
    else:
        monkeypatch.delenv("BTSEG_CHECKPOINT", raising=False)
    monkeypatch.delenv("BTSEG_MODEL_REPO", raising=False)
    spec = importlib.util.spec_from_file_location("btseg_app", ROOT / "app" / "app.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "btseg_app", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    path = tmp_path_factory.mktemp("ckpt") / "best.pt"
    torch.manual_seed(0)
    cfg = Config(run_name="tiny")
    cfg.model.kwargs = {"init_filters": 8}
    cfg.infer.roi_size = (32, 32, 32)
    torch.save({"model": build_model("segresnet", init_filters=8).state_dict(),
                "config": cfg.to_dict()}, path)
    return path


def test_run_and_viewer_end_to_end(monkeypatch, tiny_ckpt, tmp_path):
    app = load_app(monkeypatch, tiny_ckpt)
    assert not app.IS_DEMO and app.MODEL_DESCRIPTION == "local checkpoint"
    case = write_case(tmp_path, "c", shape=(40, 36, 24),
                      affine=np.diag([-1.2, -1.2, 2.0, 1.0]), with_seg=False)
    files = [case[s] for s in SEQUENCES]

    state, image, slider, show_unc, table, info, downloads, model3d = app.run(*files, True)

    assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
    assert state["labels"].shape == (40, 36, 24)  # original image space
    assert slider["maximum"] == 23 and 0 <= slider["value"] <= 23
    assert show_unc["interactive"] is True  # TTA -> uncertainty available
    assert [row[0] for row in table] == ["NCR", "ED", "ET", "TC", "WT"]
    assert "CPU" in info and "with test-time augmentation" in info

    seg_path, json_path, csv_path, glb_path = (Path(p) for p in downloads)
    assert glb_path.name == "tumor_3d.glb" and model3d == str(glb_path)
    assert glb_path.read_bytes()[:4] == b"glTF"  # at least the brain outline
    seg = nib.load(seg_path)
    assert seg.shape == (40, 36, 24)
    np.testing.assert_allclose(seg.affine, nib.load(case["flair"]).affine)
    report = json.loads(json_path.read_text())
    assert report["tta"] is True and "not for clinical use" in report["disclaimer"]
    assert set(report["volumes"]) == {"NCR", "ED", "ET", "TC", "WT"}
    rows = list(csv.reader(open(csv_path, encoding="utf-8")))
    assert rows[0] == ["region", "voxels", "volume_cm3"] and len(rows) == 6

    # viewer callbacks reuse the state and never rerun the model
    for plane, axis in (("sagittal", 0), ("coronal", 1), ("axial", 2)):
        img, update = app.change_plane(state, plane, "t1ce", 0.7, False)
        assert update["maximum"] == state["labels"].shape[axis] - 1
        assert img.shape[2] == 3
    assert app.render(state, "axial", 99, "t2", 0.3, True).dtype == np.uint8  # clipped index
    assert app.render(None, "axial", 0, "flair", 0.5, False) is None


def test_friendly_errors(monkeypatch, tiny_ckpt, tmp_path):
    app = load_app(monkeypatch, tiny_ckpt)
    a = write_case(tmp_path / "a", "a", shape=(16, 16, 16), with_seg=False)
    b = write_case(tmp_path / "b", "b", shape=(16, 16, 20), with_seg=False)
    with pytest.raises(app.gr.Error, match="Missing: T2"):
        app.run(a["flair"], a["t1"], a["t1ce"], None, False)
    with pytest.raises(app.gr.Error, match="co-registered"):
        app.run(a["flair"], a["t1"], a["t1ce"], b["t2"], False)


def test_demo_mode_without_weights_and_ui_builds(monkeypatch):
    app = load_app(monkeypatch, None)
    assert app.IS_DEMO
    demo = app.build_ui()
    assert demo is not None


def test_build_space_dry_run(tmp_path):
    script = ROOT / "scripts" / "build_space.py"
    spec = importlib.util.spec_from_file_location("build_space", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = module.main(["--space-id", "someone/space", "--model-repo", "someone/brats-segresnet",
                       "--out", str(tmp_path / "space"), "--dry-run"])
    assert {"app.py", "requirements.txt", "README.md", "model_repo.txt"} <= {
        p.name for p in out.iterdir()}
    assert (out / "btseg" / "infer.py").is_file() and not list(out.rglob("__pycache__"))
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert "  - someone/brats-segresnet" in readme and "{{" not in readme
    assert readme.startswith("---\ntitle:") and "sdk: gradio" in readme
    assert (out / "model_repo.txt").read_text().strip() == "someone/brats-segresnet"
