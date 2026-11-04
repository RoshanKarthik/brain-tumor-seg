import csv
import json

import pytest
import torch
import yaml

from btseg.train import lr_factor, main
from conftest import write_case

TINY_CONFIG = {
    "run_name": "tiny",
    "model": {"name": "segresnet", "kwargs": {"init_filters": 8}},
    "loss": {"name": "dice"},
    "data": {"roi_size": [32, 32, 32], "batch_size": 2, "num_workers": 0},
    "train": {"epochs": 1, "warmup_epochs": 1, "val_interval": 1, "lr": 1e-3},
    "infer": {"roi_size": [64, 64, 64], "sw_batch_size": 1},
}


@pytest.fixture
def run_setup(tmp_path):
    data = tmp_path / "data"
    for i in range(2):
        write_case(data / f"case{i}", f"case{i}", shape=(64, 64, 64), seed=i)
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"train": ["case0", "case1"], "val": ["case1"], "test": []}))
    config = tmp_path / "tiny.yaml"
    config.write_text(yaml.safe_dump(TINY_CONFIG))
    out = tmp_path / "run"
    args = ["--config", str(config), "--data-root", str(data), "--out", str(out),
            "--splits", str(splits)]
    return args, out


def read_log(out):
    with open(out / "train_log.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_lr_schedule_warmup_then_cosine():
    factors = [lr_factor(e, epochs=10, warmup_epochs=2) for e in range(10)]
    assert factors[0] < factors[1] < factors[2] == 1.0
    assert all(a >= b for a, b in zip(factors[2:], factors[3:], strict=False))
    assert factors[-1] < 0.05


def test_one_epoch_then_resume(run_setup):
    args, out = run_setup
    result = main(args)
    assert result["epochs_done"] == 1
    for name in ("last.pt", "best.pt", "train_log.csv", "config.json"):
        assert (out / name).exists(), name
    (row,) = read_log(out)
    assert row["epoch"] == "1"
    assert 0.0 <= float(row["val_dice_mean"]) <= 1.0

    last = torch.load(out / "last.pt", weights_only=True)
    best = torch.load(out / "best.pt", weights_only=True)
    assert last["epoch"] == 1 and best["epoch"] == 1
    assert set(last) >= {"model", "optimizer", "scheduler", "scaler", "best_metric"}
    assert best["config"]["model"]["name"] == "segresnet"

    with pytest.raises(FileExistsError):
        main(args)  # refuses to overwrite a run without --resume

    result = main([*args, "--resume", "--epochs", "2"])
    assert result["epochs_done"] == 2
    assert [r["epoch"] for r in read_log(out)] == ["1", "2"]
    resumed = torch.load(out / "last.pt", weights_only=True)
    assert resumed["epoch"] == 2
    assert resumed["scheduler"]["last_epoch"] == 2
    w1, w2 = last["model"], resumed["model"]
    assert any(not torch.equal(w1[k], w2[k]) for k in w1), "weights did not change"


def test_resume_requires_last_checkpoint(run_setup):
    args, _ = run_setup
    with pytest.raises(FileNotFoundError, match="last.pt"):
        main([*args, "--resume"])


def test_time_limit_stops_before_starting_an_epoch(run_setup):
    args, out = run_setup
    result = main([*args, "--time-limit-hours", "0"])
    assert result["epochs_done"] == 0
    assert not (out / "last.pt").exists()


def test_missing_splits_file_fails(run_setup, tmp_path):
    args, _ = run_setup
    args[args.index("--splits") + 1] = str(tmp_path / "nope.json")
    with pytest.raises(FileNotFoundError, match="make_splits"):
        main(args)
