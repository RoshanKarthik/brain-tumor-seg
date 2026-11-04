import importlib.util
from pathlib import Path

import pytest

from btseg.data.dataset import (
    build_dataset,
    case_to_item,
    load_splits,
    make_splits,
    save_splits,
    select_split,
)
from btseg.data.discover import SEQUENCES, discover_cases
from btseg.data.transforms import get_val_transforms
from conftest import write_case

ROOT = Path(__file__).resolve().parents[1]
IDS = [f"BraTS2021_{i:05d}" for i in range(1251)]


def test_split_sizes_disjoint_and_complete():
    s = make_splits(IDS)
    assert (len(s["train"]), len(s["val"]), len(s["test"])) == (1001, 125, 125)
    train, val, test = map(set, (s["train"], s["val"], s["test"]))
    assert not (train & val or train & test or val & test)
    assert train | val | test == set(IDS)
    assert s["seed"] == 42 and s["n_cases"] == 1251


def test_split_is_deterministic_and_order_independent():
    assert make_splits(IDS) == make_splits(list(reversed(IDS)))
    assert make_splits(IDS, seed=1)["test"] != make_splits(IDS)["test"]


def test_split_rejects_bad_input():
    with pytest.raises(ValueError):
        make_splits(["a", "a", "b"])
    with pytest.raises(ValueError):
        make_splits(IDS, fractions=(0.8, 0.1, 0.2))


def test_save_load_and_select(tmp_path):
    cases = [{"id": i} for i in IDS[:20]]
    s = make_splits([c["id"] for c in cases])
    save_splits(s, tmp_path / "sub" / "splits.json")
    loaded = load_splits(tmp_path / "sub" / "splits.json")
    assert loaded == s
    assert [c["id"] for c in select_split(cases, loaded, "val")] == s["val"]
    with pytest.raises(KeyError):
        select_split(cases[:5], loaded, "train")
    with pytest.raises(ValueError):
        select_split(cases, loaded, "holdout")


def test_case_to_item_orders_sequences(synthetic_case):
    item = case_to_item(synthetic_case)
    assert item["image"] == [synthetic_case[s] for s in SEQUENCES]
    assert item["label"] == synthetic_case["seg"]
    assert "label" not in case_to_item({k: v for k, v in synthetic_case.items() if k != "seg"})


def test_build_dataset_end_to_end(tmp_path):
    for i in range(2):
        write_case(tmp_path / f"c{i}", f"case{i}", shape=(32, 32, 32), seed=i)
    cases = discover_cases(tmp_path, require_seg=True)
    ds = build_dataset(cases, get_val_transforms())
    assert len(ds) == 2
    item = ds[1]
    assert item["id"] == "case1"
    assert item["image"].shape == (4, 32, 32, 32)
    assert item["label"].shape == (3, 32, 32, 32)


def test_make_splits_script(tmp_path):
    for i in range(10):
        d = tmp_path / "data" / f"case{i:02d}"
        d.mkdir(parents=True)
        for part in (*SEQUENCES, "seg"):
            (d / f"case{i:02d}_{part}.nii.gz").touch()
    script = ROOT / "scripts" / "make_splits.py"
    spec = importlib.util.spec_from_file_location("make_splits", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    out = tmp_path / "splits.json"
    splits = module.main(["--data-root", str(tmp_path / "data"), "--out", str(out)])
    assert load_splits(out) == splits
    assert (len(splits["train"]), len(splits["val"]), len(splits["test"])) == (8, 1, 1)
