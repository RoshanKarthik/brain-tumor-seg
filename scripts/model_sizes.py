"""Count parameters of every model in configs/*.yaml -> results/model_sizes.json.

    python scripts/model_sizes.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import monai
import torch

from btseg.config import load_config
from btseg.models.factory import build_model, count_parameters

ROOT = Path(__file__).resolve().parents[1]


def compute_model_sizes(config_dir: Path) -> dict:
    models = {}
    for path in sorted(config_dir.glob("*.yaml")):
        cfg = load_config(path)
        model = build_model(cfg.model.name, **cfg.model.kwargs)
        models[cfg.model.name] = {
            "config": path.relative_to(config_dir.parent).as_posix(),
            "trainable_parameters": count_parameters(model),
            "total_parameters": count_parameters(model, trainable_only=False),
        }
    return {"torch": torch.__version__.split("+")[0], "monai": monai.__version__, "models": models}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", type=Path, default=ROOT / "configs")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "model_sizes.json")
    args = parser.parse_args(argv)

    sizes = compute_model_sizes(args.configs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(sizes, indent=2) + "\n", encoding="utf-8")
    for name, info in sizes["models"].items():
        print(f"{name:10s} {info['trainable_parameters']:>12,d} parameters  ({info['config']})")
    return sizes


if __name__ == "__main__":
    main()
