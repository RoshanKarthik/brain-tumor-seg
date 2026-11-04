"""Predictor: 4 NIfTI files -> label map + region probabilities in the original image space.

    python -m btseg.infer --ckpt runs/segresnet_dice/best.pt --case-dir path/to/case [--tta]

``--ckpt`` is a local checkpoint or a Hugging Face model repo id (then ``--filename``
selects the file inside it). Writes ``seg.nii.gz`` (BraTS labels, aligned with the
input files) and ``volumes.json`` to ``--out`` (default: the case directory).

The inputs go through the same orientation / spacing / normalization as in training;
the predicted probabilities are mapped back with MONAI's inverse transforms, so the
output has the shape and affine of the uploaded FLAIR. Optional test-time augmentation
(TTA) averages the 8 flip combinations and reports the per-voxel variance as an
uncertainty map.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import MetaTensor
from monai.inferers import sliding_window_inference
from torch import nn

from btseg.config import Config, config_from_dict
from btseg.data.discover import SEQUENCES, discover_cases
from btseg.data.labels import regions_to_labels
from btseg.data.transforms import get_infer_transforms
from btseg.models.factory import build_model
from btseg.postprocess import postprocess
from btseg.volumetrics import region_volumes, voxel_volume_mm3

FLIP_DIMS = [dims for k in range(4) for dims in itertools.combinations((2, 3, 4), k)]


@dataclass
class PredictionResult:
    label_map: np.ndarray  # (H, W, D) uint8 BraTS labels in the input image space
    probabilities: np.ndarray  # (3, H, W, D) float32, [TC, WT, ET]
    uncertainty: np.ndarray | None  # (3, H, W, D) float32 variance over TTA flips
    affine: np.ndarray  # 4x4 affine of the input FLAIR
    volumes: dict[str, dict[str, float]]  # per region: voxels, cm3
    inference_seconds: float
    device: str
    tta: bool


def validate_inputs(paths: Sequence[str | Path], atol: float = 1e-3) -> None:
    """Fail with a readable message unless all inputs are 3D with equal shape and affine."""
    if len(paths) != len(SEQUENCES):
        raise ValueError(
            f"Expected {len(SEQUENCES)} files ({', '.join(SEQUENCES)}), got {len(paths)}"
        )
    images = {}
    for name, path in zip(SEQUENCES, paths, strict=True):
        try:
            img = nib.load(str(path))
        except Exception as exc:
            raise ValueError(
                f"{name.upper()}: cannot read {Path(path).name} as NIfTI ({exc})"
            ) from exc
        shape = tuple(int(s) for s in img.shape)
        if len(shape) != 3:
            raise ValueError(f"{name.upper()}: expected a 3D volume, got shape {shape}")
        images[name] = (shape, img.affine)
    ref_shape, ref_affine = images[SEQUENCES[0]]
    for name, (shape, affine) in images.items():
        if shape != ref_shape:
            raise ValueError(
                f"{name.upper()} has shape {shape} but FLAIR has {ref_shape}; "
                "the 4 sequences must be co-registered"
            )
        if not np.allclose(affine, ref_affine, atol=atol):
            raise ValueError(
                f"{name.upper()} has a different affine than FLAIR; "
                "the 4 sequences must be co-registered"
            )


class Predictor:
    def __init__(self, model: nn.Module, config: Config, device: str | torch.device = "auto"):
        self.model = model.eval()
        self.config = config
        self.transforms = get_infer_transforms(config.data.pixdim)
        self.to(device)

    @classmethod
    def from_checkpoint(
        cls, source: str | Path, device: str | torch.device = "auto", filename: str = "best.pt"
    ) -> Predictor:
        """Load ``best.pt`` (weights + config) from a path or a Hugging Face model repo."""
        path = Path(source)
        if not path.is_file():
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(repo_id=str(source), filename=filename))
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        config = config_from_dict(ckpt["config"])
        model = build_model(config.model.name, **config.model.kwargs)
        model.load_state_dict(ckpt["model"])
        return cls(model, config, device)

    def to(self, device: str | torch.device) -> Predictor:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    @torch.no_grad()
    def _probabilities(self, x: torch.Tensor, flip_dims: tuple[int, ...]) -> torch.Tensor:
        cuda = self.device.type == "cuda"
        x_in = torch.flip(x, flip_dims) if flip_dims else x
        with torch.autocast(device_type=self.device.type, enabled=cuda):
            logits = sliding_window_inference(
                x_in,
                roi_size=self.config.infer.roi_size,
                sw_batch_size=self.config.infer.sw_batch_size if cuda else 1,
                predictor=self.model,
                overlap=self.config.infer.overlap,
                mode=self.config.infer.mode,
            )
        logits = torch.flip(logits, flip_dims) if flip_dims else logits
        return torch.sigmoid(logits.float())

    def predict(self, flair, t1, t1ce, t2, tta: bool = False) -> PredictionResult:
        paths = [str(p) for p in (flair, t1, t1ce, t2)]
        validate_inputs(paths)
        ref = nib.load(paths[0])
        start = time.perf_counter()

        image = self.transforms({"image": paths})["image"]  # MetaTensor (4, H', W', D')
        x = image.as_tensor().unsqueeze(0).to(self.device)

        # running mean / variance (Welford) over the flip combinations
        mean, m2 = None, None
        flips = FLIP_DIMS if tta else [()]
        for k, dims in enumerate(flips, start=1):
            p = self._probabilities(x, dims)[0]
            if mean is None:
                mean, m2 = p.clone(), torch.zeros_like(p)
            else:
                delta = p - mean
                mean += delta / k
                m2 += delta * (p - mean)
        maps = torch.cat([mean, m2 / len(flips)]) if tta else mean

        # back to the input space: undo spacing and orientation (bilinear for probabilities)
        pred = MetaTensor(maps.cpu(), meta=dict(image.meta))
        pred.applied_operations = list(image.applied_operations)
        inverted = self.transforms.inverse({"image": pred})["image"]
        maps = np.clip(np.asarray(inverted.detach().cpu(), dtype=np.float32), 0.0, None)
        if maps.shape[1:] != tuple(ref.shape):
            raise RuntimeError(f"Inverse mapping gave {maps.shape[1:]}, input is {ref.shape}")

        probabilities = np.minimum(maps[:3], 1.0)
        uncertainty = maps[3:] if tta else None
        regions = postprocess(probabilities, self.config.postprocess)
        label_map = np.asarray(regions_to_labels(regions, self.config.data.et_label), np.uint8)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return PredictionResult(
            label_map=label_map,
            probabilities=probabilities,
            uncertainty=uncertainty,
            affine=ref.affine.copy(),
            volumes=region_volumes(label_map, ref.affine, self.config.data.et_label),
            inference_seconds=time.perf_counter() - start,
            device=str(self.device),
            tta=tta,
        )


def save_label_map(label_map: np.ndarray, reference: str | Path, path: str | Path) -> None:
    """Save labels as uint8 NIfTI with the reference image's affine and orientation codes."""
    ref = nib.load(str(reference))
    header = ref.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1.0, 0.0)
    nib.save(nib.Nifti1Image(label_map.astype(np.uint8), ref.affine, header), str(path))


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Segment one case and write seg.nii.gz + volumes.json")
    p.add_argument("--ckpt", required=True, help="best.pt path or Hugging Face model repo id")
    p.add_argument("--filename", default="best.pt", help="File inside the HF repo")
    p.add_argument("--case-dir", required=True, type=Path)
    p.add_argument("--out", type=Path, help="Output directory (default: --case-dir)")
    p.add_argument("--tta", action="store_true", help="8-flip test-time augmentation")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    cases = discover_cases(args.case_dir)
    if len(cases) != 1:
        raise ValueError(f"Expected exactly one case in {args.case_dir}, found {len(cases)}")
    case = cases[0]
    predictor = Predictor.from_checkpoint(args.ckpt, device=args.device, filename=args.filename)
    result = predictor.predict(*(case[s] for s in SEQUENCES), tta=args.tta)

    out = args.out or args.case_dir
    out.mkdir(parents=True, exist_ok=True)
    save_label_map(result.label_map, case["flair"], out / "seg.nii.gz")
    report = {
        "case_id": case["id"],
        "checkpoint": str(args.ckpt),
        "model": predictor.config.model.name,
        "tta": result.tta,
        "device": result.device,
        "inference_seconds": round(result.inference_seconds, 3),
        "voxel_volume_mm3": voxel_volume_mm3(result.affine),
        "et_label": predictor.config.data.et_label,
        "volumes": result.volumes,
    }
    (out / "volumes.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{case['id']}: " + ", ".join(
        f"{r} {v['cm3']:.2f} cm3" for r, v in result.volumes.items()
    ) + f" ({result.inference_seconds:.1f}s on {result.device}) -> {out}")
    return report


if __name__ == "__main__":
    main()
