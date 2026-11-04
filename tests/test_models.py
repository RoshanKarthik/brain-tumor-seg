import importlib.util
import json
from pathlib import Path

import pytest
import torch

from btseg.config import load_config
from btseg.models.factory import (
    MODEL_DEFAULTS,
    build_model,
    check_roi_size,
    count_parameters,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", sorted(MODEL_DEFAULTS))
def test_forward_shape(name):
    torch.manual_seed(0)
    model = build_model(name).eval()
    x = torch.randn(1, 4, 64, 64, 64)
    with torch.no_grad():
        y = model(x)
    print(f"{name}: {count_parameters(model):,} parameters")
    assert y.shape == (1, 3, 64, 64, 64)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("config", ["segresnet", "unet3d", "swinunetr"])
def test_config_builds_the_same_model_as_defaults(config):
    cfg = load_config(ROOT / "configs" / f"{config}.yaml")
    from_config = build_model(cfg.model.name, **cfg.model.kwargs)
    assert count_parameters(from_config) == count_parameters(build_model(cfg.model.name))
    check_roi_size(cfg.model.name, cfg.data.roi_size)
    check_roi_size(cfg.model.name, cfg.infer.roi_size)


def test_unet_baseline_size_is_comparable_to_segresnet():
    seg = count_parameters(build_model("segresnet"))
    unet = count_parameters(build_model("unet3d"))
    assert abs(unet - seg) / seg < 0.1


def test_kwargs_override_defaults():
    small = build_model("segresnet", init_filters=8)
    assert count_parameters(small) < count_parameters(build_model("segresnet"))


def test_unknown_model_and_bad_roi_raise():
    with pytest.raises(ValueError, match="Unknown model"):
        build_model("resnet50")
    with pytest.raises(ValueError, match="divisible by 32"):
        check_roi_size("swinunetr", (96, 96, 80))


def test_committed_model_sizes_are_current():
    script = ROOT / "scripts" / "model_sizes.py"
    spec = importlib.util.spec_from_file_location("model_sizes", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    committed = json.loads((ROOT / "results" / "model_sizes.json").read_text(encoding="utf-8"))
    current = module.compute_model_sizes(ROOT / "configs")
    assert committed["models"] == current["models"], "rerun: python scripts/model_sizes.py"
