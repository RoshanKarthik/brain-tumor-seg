"""ONNX export + PyTorch parity check, and an ONNX Runtime model for CPU inference.

    python -m btseg.export --ckpt runs/segresnet_dice/best.pt [--out model.onnx]

The network is exported for a fixed patch size (``infer.roi_size``, 128^3 by default) with
a dynamic batch dimension; sliding-window inference over whole volumes stays in Python
(MONAI), so :class:`OnnxModel` drops into :class:`btseg.infer.Predictor` unchanged.
Next to ``model.onnx`` the export writes ``export_info.json`` with the config, the patch
size and the measured parity (max |ONNX - PyTorch| on a random input, must be < 1e-3).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from btseg.config import config_from_dict
from btseg.infer import Predictor

PARITY_TOLERANCE = 1e-3


def export_onnx(
    model: nn.Module, path: str | Path, roi_size=(128, 128, 128), opset: int = 18
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    example = torch.zeros(1, 4, *roi_size)
    program = torch.onnx.export(
        model,
        (example,),
        dynamo=True,
        opset_version=opset,
        verbose=False,  # its progress messages contain emoji that break cp1252 consoles
        input_names=["image"],
        output_names=["logits"],
        dynamic_shapes=({0: torch.export.Dim("batch", min=1, max=16)},),
    )
    program.save(str(path))
    return path


class OnnxModel(nn.Module):
    """ONNX Runtime (CPU) model with the PyTorch call signature: (B, 4, *roi) -> logits."""

    def __init__(self, path: str | Path, threads: int | None = None):
        super().__init__()
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        array = x.detach().cpu().numpy().astype(np.float32, copy=False)
        return torch.from_numpy(self.session.run(None, {self.input_name: array})[0])


def parity(model: nn.Module, onnx_path: str | Path, roi_size, batch: int = 1, seed: int = 0):
    """Max absolute difference between PyTorch and ONNX Runtime logits on a random input."""
    x = torch.randn(batch, 4, *roi_size, generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        ref = model.eval().cpu()(x)
        out = OnnxModel(onnx_path)(x)
    return float((out - ref).abs().max())


def load_onnx_predictor(onnx_path: str | Path, threads: int | None = None) -> Predictor:
    """CPU Predictor from ``model.onnx`` + its ``export_info.json``."""
    onnx_path = Path(onnx_path)
    info = json.loads((onnx_path.parent / "export_info.json").read_text(encoding="utf-8"))
    return Predictor(OnnxModel(onnx_path, threads), config_from_dict(info["config"]), "cpu")


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Export a checkpoint to ONNX and check parity.")
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--out", type=Path, help="Default: model.onnx next to the checkpoint")
    p.add_argument("--opset", type=int, default=18)
    args = p.parse_args(argv)

    predictor = Predictor.from_checkpoint(args.ckpt, device="cpu")
    cfg = predictor.config
    roi = tuple(cfg.infer.roi_size)
    out = args.out or args.ckpt.with_name("model.onnx")
    export_onnx(predictor.model, out, roi, args.opset)
    diff = parity(predictor.model, out, roi, batch=2)
    info = {
        "source_checkpoint": args.ckpt.name,
        "model": cfg.model.name,
        "roi_size": list(roi),
        "opset": args.opset,
        "parity_max_abs_diff": diff,
        "parity_tolerance": PARITY_TOLERANCE,
        "torch": torch.__version__,
        "onnxruntime": ort.__version__,
        "config": cfg.to_dict(),
    }
    out.with_name("export_info.json").write_text(json.dumps(info, indent=2) + "\n", "utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB); parity max |diff| = {diff:.2e}")
    if not diff < PARITY_TOLERANCE:
        raise SystemExit(f"Parity check failed: {diff:.2e} >= {PARITY_TOLERANCE}")
    return info


if __name__ == "__main__":
    main()
