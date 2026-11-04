"""Shared helpers: tiny synthetic BraTS-like cases written to a temp dir."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from btseg.data.discover import SEQUENCES


def synthetic_label_map(shape=(64, 64, 64), et_label: int = 4) -> np.ndarray:
    """Concentric tumor: NCR core, ET shell, ED rim, inside a cube volume."""
    grid = np.indices(shape, dtype=np.float32)
    center = np.array(shape, dtype=np.float32)[:, None, None, None] * np.array([0.55, 0.5, 0.45])[
        :, None, None, None
    ]
    dist = np.sqrt(((grid - center) ** 2).sum(0))
    r = min(shape) * 0.25
    seg = np.zeros(shape, dtype=np.uint8)
    seg[dist < r] = 2
    seg[dist < 0.6 * r] = et_label
    seg[dist < 0.35 * r] = 1
    return seg


def write_case(
    directory: Path,
    case_id: str,
    shape=(64, 64, 64),
    spacing=(1.0, 1.0, 1.0),
    et_label: int = 4,
    seed: int = 0,
    with_seg: bool = True,
    affine: np.ndarray | None = None,
) -> dict[str, str]:
    """Write 4 sequences (+ seg) as NIfTI and return the case dict ``discover_cases`` would."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    affine = np.diag([*spacing, 1.0]) if affine is None else affine

    grid = np.indices(shape, dtype=np.float32)
    center = (np.array(shape, dtype=np.float32) - 1)[:, None, None, None] / 2
    brain = (((grid - center) / (np.array(shape)[:, None, None, None] * 0.42)) ** 2).sum(0) < 1
    seg = synthetic_label_map(shape, et_label) * brain

    case = {"id": case_id}
    for k, seq in enumerate(SEQUENCES):
        img = (100.0 + 20.0 * k + 30.0 * (seg > 0) + rng.normal(0, 5, shape)) * brain
        path = directory / f"{case_id}_{seq}.nii.gz"
        nib.save(nib.Nifti1Image(img.astype(np.float32), affine), path)
        case[seq] = str(path)
    if with_seg:
        path = directory / f"{case_id}_seg.nii.gz"
        nib.save(nib.Nifti1Image(seg.astype(np.uint8), affine), path)
        case["seg"] = str(path)
    return case


@pytest.fixture
def synthetic_case(tmp_path) -> dict[str, str]:
    return write_case(tmp_path / "BraTS2021_00000", "BraTS2021_00000")
