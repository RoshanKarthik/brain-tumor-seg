import csv
import json

import numpy as np
import pytest
import torch

from btseg.config import Config
from btseg.evaluate import (
    describe,
    evaluate,
    main,
    pick_overlay_cases,
    volume_agreement,
)
from btseg.infer import Predictor
from btseg.models.factory import build_model
from conftest import write_case
from test_infer import ThresholdModel

FIGURES = ("dice_boxplot.png", "bland_altman.png", "volume_scatter.png", "overlays.png")


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def mini_split(tmp_path):
    root = tmp_path / "data"
    cases = [write_case(root / f"case{i}", f"case{i}", shape=(40, 40, 32), seed=i)
             for i in range(3)]
    return root, cases


def test_evaluate_writes_results_and_ablation(mini_split, tmp_path):
    _, cases = mini_split
    out = tmp_path / "results" / "threshold_model"
    summary = evaluate(Predictor(ThresholdModel(), Config(), device="cpu"), cases, out,
                       ablation=True, provenance={"run_name": "threshold_model"})

    for name in ("per_case.csv", "summary.json", "ablation.csv", *FIGURES):
        assert (out / name).stat().st_size > 0, name
    rows = read_csv(out / "per_case.csv")
    assert [r["case_id"] for r in rows] == ["case0", "case1", "case2"]
    assert {"dice_WT", "hd95_ET", "vol_pred_NCR", "vol_gt_WT"} <= set(rows[0])
    # the threshold "model" finds the whole tumor, so WT is near perfect
    assert summary["dice"]["WT"]["mean"] > 0.95
    assert summary["dice"]["WT"]["n"] == 3
    assert summary["volume_agreement"]["WT"]["mean_abs_percent_error"] < 10
    assert json.loads((out / "summary.json").read_text())["run_name"] == "threshold_model"
    assert len(list((out / "predictions").glob("*_seg.nii.gz"))) == 3

    ablation = read_csv(out / "ablation.csv")
    assert [a["variant"] for a in ablation] == [
        "no_tta+threshold_only", "no_tta+postprocess", "tta+threshold_only", "tta+postprocess",
    ]
    # thresholding commutes with flips, so TTA cannot change this model's output
    assert ablation[0]["dice_WT_mean"] == ablation[2]["dice_WT_mean"]


def test_cli_with_checkpoint_and_split(mini_split, tmp_path):
    root, _ = mini_split
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"train": [], "val": ["case0"], "test": ["case1", "case2"]}))
    torch.manual_seed(0)
    cfg = Config(run_name="tiny")
    cfg.model.kwargs = {"init_filters": 8}
    ckpt = tmp_path / "best.pt"
    torch.save({"model": build_model("segresnet", init_filters=8).state_dict(),
                "config": cfg.to_dict()}, ckpt)

    common = ["--ckpt", str(ckpt), "--data-root", str(root), "--splits", str(splits),
              "--out-root", str(tmp_path / "results"), "--device", "cpu"]
    summary = main([*common, "--split", "test"])
    assert summary["run_name"] == "tiny" and summary["split"] == "test"
    assert summary["n_cases"] == 2
    for r in ("WT", "TC", "ET"):
        assert 0.0 <= summary["dice"][r]["mean"] <= 1.0
        assert 0.0 <= summary["hd95_mm"][r]["mean"] <= 373.13
    assert (tmp_path / "results" / "tiny" / "overlays.png").exists()

    val = main([*common, "--split", "val"])
    assert val["run_name"] == "tiny_val" and val["n_cases"] == 1
    assert val["dice"]["WT"]["std"] is None  # one case: no std, JSON stays valid
    json.loads((tmp_path / "results" / "tiny_val" / "summary.json").read_text())


def test_describe_and_volume_agreement():
    d = describe([0.5, 0.7, 0.9])
    assert d["mean"] == pytest.approx(0.7) and d["median"] == pytest.approx(0.7)
    assert d["std"] == pytest.approx(0.2) and d["n"] == 3

    gt = np.array([10.0, 20.0, 0.0, 40.0])
    pred = np.array([12.0, 18.0, 1.0, 40.0])
    va = volume_agreement(pred, gt)
    assert va["mae_cm3"] == pytest.approx(1.25)
    assert va["bias_cm3"] == pytest.approx(0.25)
    # % error only over cases with a nonzero ground truth: (20% + 10% + 0%) / 3
    assert va["mean_abs_percent_error"] == pytest.approx(10.0)
    assert va["percent_error_n"] == 3
    assert va["pearson_r"] > 0.95
    lo, hi = va["loa_cm3"]
    assert lo < va["bias_cm3"] < hi
    assert volume_agreement([0.0, 0.0], [0.0, 0.0])["pearson_r"] is None


def test_pick_overlay_cases():
    picks = pick_overlay_cases([0.1, 0.9, 0.5, 0.8, 0.3, 0.6, 0.95])
    labels = [label for label, _ in picks]
    idx = [i for _, i in picks]
    assert labels == ["best", "best", "median", "median", "worst", "worst"]
    assert idx[:2] == [6, 1] and idx[4:] == [0, 4]
    assert len(set(idx)) == 6
    assert len(pick_overlay_cases([0.5, 0.6])) == 2  # fewer cases than slots, no repeats


def test_pp_grid_and_postprocess_overrides(mini_split, tmp_path):
    root, _ = mini_split
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"train": [], "val": ["case0", "case1"], "test": ["case2"]}))
    cfg = Config(run_name="tiny")
    cfg.model.kwargs = {"init_filters": 8}
    ckpt = tmp_path / "best.pt"
    torch.manual_seed(0)
    torch.save({"model": build_model("segresnet", init_filters=8).state_dict(),
                "config": cfg.to_dict()}, ckpt)
    common = ["--ckpt", str(ckpt), "--data-root", str(root), "--splits", str(splits),
              "--out-root", str(tmp_path / "results"), "--device", "cpu"]

    main([*common, "--split", "val", "--pp-grid"])
    grid = read_csv(tmp_path / "results" / "tiny_val" / "pp_grid.csv")
    assert len(grid) == 4 * 5
    assert {(int(g["min_component_voxels"]), int(g["et_min_voxels"])) for g in grid} >= {(0, 0)}
    means = [float(g["dice_mean_of_regions"]) for g in grid]
    assert means == sorted(means, reverse=True)

    summary = main([*common, "--split", "test", "--min-component-voxels", "200",
                    "--et-min-voxels", "300"])
    assert summary["postprocess"]["min_component_voxels"] == 200
    assert summary["postprocess"]["et_min_voxels"] == 300


def test_grid_matches_postprocess():
    from btseg.config import PostprocessConfig
    from btseg.evaluate import grid_dice
    from btseg.metrics import dice
    from btseg.postprocess import postprocess

    rng = np.random.default_rng(0)
    probs = rng.random((3, 24, 24, 24)).astype(np.float32) ** 3
    gt = probs > 0.4
    results = grid_dice(probs, gt, 0.5)
    for (mc, em), values in results.items():
        r = postprocess(probs, PostprocessConfig(min_component_voxels=mc, et_min_voxels=em))
        expected = [dice(r[i], gt[i]) for i in (1, 0, 2)]
        assert values == pytest.approx(expected)
