from pathlib import Path

import pytest

from btseg.config import Config, config_from_dict, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_main_config_matches_plan():
    cfg = load_config(ROOT / "configs" / "segresnet.yaml")
    assert cfg.model.name == "segresnet"
    assert cfg.model.kwargs["blocks_down"] == [1, 2, 2, 4]
    assert cfg.loss.name == "dice"
    assert cfg.data.roi_size == (128, 128, 128)
    assert cfg.data.et_label == 4
    assert cfg.train.lr == 1e-4 and cfg.train.weight_decay == 1e-5
    assert cfg.infer.overlap == 0.5


def test_all_configs_load():
    paths = sorted((ROOT / "configs").glob("*.yaml"))
    assert len(paths) >= 2
    for path in paths:
        assert isinstance(load_config(path), Config)


def test_defaults_and_round_trip():
    cfg = config_from_dict({"model": {"name": "unet3d"}})
    assert cfg.loss.name == "dice" and cfg.data.batch_size == 2
    assert config_from_dict(cfg.to_dict()) == cfg


@pytest.mark.parametrize(
    "raw", [{"modle": {}}, {"train": {"learning_rate": 1e-3}}, {"data": {"roi": [96, 96, 96]}}]
)
def test_unknown_keys_raise(raw):
    with pytest.raises(ValueError, match="Unknown key"):
        config_from_dict(raw)
