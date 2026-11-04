import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from btseg.config import Config
from btseg.data.discover import SEQUENCES
from btseg.export import PARITY_TOLERANCE, export_onnx, load_onnx_predictor, main, parity
from btseg.infer import Predictor
from btseg.models.factory import build_model
from conftest import write_case

ROOT = Path(__file__).resolve().parents[1]
ROI = (64, 64, 64)


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    torch.manual_seed(0)
    cfg = Config(run_name="tiny")
    cfg.model.kwargs = {"init_filters": 8}
    cfg.infer.roi_size = ROI
    model = build_model("segresnet", init_filters=8)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "epoch": 7,
                "val_dice": {"TC": 0.5, "WT": 0.6, "ET": 0.4}, "val_dice_mean": 0.5},
               d / "best.pt")
    (d / "config.json").write_text(json.dumps(cfg.to_dict()))
    (d / "train_log.csv").write_text("epoch,train_loss\n1,0.9\n")
    return d


def test_export_cli_parity_and_info(tiny_checkpoint):
    info = main(["--ckpt", str(tiny_checkpoint / "best.pt")])
    assert (tiny_checkpoint / "model.onnx").stat().st_size > 0
    saved = json.loads((tiny_checkpoint / "export_info.json").read_text())
    assert saved["parity_max_abs_diff"] == info["parity_max_abs_diff"] < PARITY_TOLERANCE
    assert saved["roi_size"] == list(ROI) and saved["config"]["model"]["name"] == "segresnet"


def test_parity_detects_a_different_model(tiny_checkpoint, tmp_path):
    torch.manual_seed(1)
    other = build_model("segresnet", init_filters=8)
    path = export_onnx(other, tmp_path / "other.onnx", ROI)
    torch.manual_seed(0)
    original = build_model("segresnet", init_filters=8)
    assert parity(other, path, ROI) < PARITY_TOLERANCE
    assert parity(original, path, ROI) > PARITY_TOLERANCE


def test_onnx_predictor_matches_pytorch_predictor(tiny_checkpoint, tmp_path):
    if not (tiny_checkpoint / "model.onnx").exists():
        main(["--ckpt", str(tiny_checkpoint / "best.pt")])
    case = write_case(tmp_path, "c", shape=(48, 48, 40), with_seg=False)
    files = [case[s] for s in SEQUENCES]
    torch_result = Predictor.from_checkpoint(tiny_checkpoint / "best.pt", device="cpu")
    onnx_result = load_onnx_predictor(tiny_checkpoint / "model.onnx")
    a = torch_result.predict_maps(files).plain
    b = onnx_result.predict_maps(files).plain
    assert np.abs(a - b).max() < PARITY_TOLERANCE


def load_upload_script():
    spec = importlib.util.spec_from_file_location("upload_weights",
                                                  ROOT / "scripts" / "upload_weights.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_summary():
    stat = {"mean": 0.9, "std": 0.05, "median": 0.92, "n": 125}
    hd = {"mean": 4.25, "std": 3.0, "median": 3.1, "n": 125}
    vol = {"mae_cm3": 3.21, "mean_abs_percent_error": 7.5, "percent_error_n": 120,
           "pearson_r": 0.987, "bias_cm3": -0.42, "loa_cm3": [-8.0, 7.0]}
    return {"split": "test", "n_cases": 125, "tta": True, "postprocess": {"threshold": 0.5},
            "metric_note": "BraTS conventions.",
            "dice": {r: stat for r in ("WT", "TC", "ET")},
            "hd95_mm": {r: hd for r in ("WT", "TC", "ET")},
            "volume_agreement": {r: vol for r in ("NCR", "ED", "ET", "TC", "WT")}}


def test_upload_dry_run_stages_files_and_card(tiny_checkpoint, tmp_path, monkeypatch):
    import huggingface_hub

    def no_network(*args, **kwargs):
        raise AssertionError("dry run must not contact the Hub")

    monkeypatch.setattr(huggingface_hub, "HfApi", no_network)
    results = tmp_path / "results"
    results.mkdir()
    (results / "summary.json").write_text(json.dumps(fake_summary()))
    (results / "dice_boxplot.png").write_bytes(b"png")
    module = load_upload_script()

    staged = module.main(["--repo-id", "someone/brats-test", "--run-dir", str(tiny_checkpoint),
                          "--results", str(results), "--staging", str(tmp_path / "stage"),
                          "--dry-run"])
    names = {p.relative_to(staged).as_posix() for p in staged.rglob("*") if p.is_file()}
    assert {"best.pt", "config.json", "train_log.csv", "README.md",
            "results/summary.json", "results/dice_boxplot.png"} <= names
    card = (staged / "README.md").read_text(encoding="utf-8")
    assert "Not a medical device" in card and "not for clinical use" in card
    assert "| WT | 0.900 ± 0.050 | 0.920 | 4.25 | 3.10 |" in card
    assert "3.21" in card and "0.987" in card  # volume agreement from summary.json
    assert "epoch 7, mean validation Dice 0.500" in card
    assert 'Predictor.from_checkpoint("someone/brats-test")' in card
    n_params = sum(p.numel() for p in build_model("segresnet", init_filters=8).parameters())
    assert f"{n_params:,} parameters" in card


def test_card_says_tbd_without_results(tiny_checkpoint, tmp_path):
    module = load_upload_script()
    staged = module.main(["--repo-id", "someone/x", "--run-dir", str(tiny_checkpoint),
                          "--staging", str(tmp_path / "stage"), "--dry-run"])
    card = (staged / "README.md").read_text(encoding="utf-8")
    assert "TBD: not evaluated on the test split yet." in card
    assert "| WT |" not in card
