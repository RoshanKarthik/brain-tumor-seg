"""YAML config -> dataclass.

Unknown keys are errors, so a typo in a config file fails instead of being ignored.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    name: str = "segresnet"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LossConfig:
    name: str = "dice"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    roi_size: tuple[int, int, int] = (128, 128, 128)
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0)
    et_label: int = 4
    batch_size: int = 2
    num_workers: int = 4
    splits: str = "data/splits.json"


@dataclass
class TrainConfig:
    epochs: int = 150
    lr: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    val_interval: int = 5
    amp: bool = True
    seed: int = 42


@dataclass
class InferConfig:
    roi_size: tuple[int, int, int] = (128, 128, 128)
    overlap: float = 0.5
    sw_batch_size: int = 2
    mode: str = "gaussian"  # sliding-window blending: gaussian | constant


@dataclass
class Config:
    run_name: str = "run"
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_SECTIONS = {
    "model": ModelConfig,
    "loss": LossConfig,
    "data": DataConfig,
    "train": TrainConfig,
    "infer": InferConfig,
}
_TUPLE_FIELDS = {("data", "roi_size"), ("data", "pixdim"), ("infer", "roi_size")}


def _check_keys(where: str, given: dict, cls) -> None:
    allowed = {f.name for f in dataclasses.fields(cls)}
    unknown = set(given) - allowed
    if unknown:
        raise ValueError(
            f"Unknown key(s) in {where}: {sorted(unknown)}; allowed: {sorted(allowed)}"
        )


def config_from_dict(raw: dict[str, Any]) -> Config:
    raw = dict(raw or {})
    _check_keys("config", raw, Config)
    sections = {}
    for name, cls in _SECTIONS.items():
        values = dict(raw.pop(name, None) or {})
        _check_keys(name, values, cls)
        for key in values:
            if (name, key) in _TUPLE_FIELDS:
                values[key] = tuple(values[key])
        sections[name] = cls(**values)
    return Config(**raw, **sections)


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        return config_from_dict(yaml.safe_load(f))
