"""Disk cache of the deterministic part of the training pipeline (speed only).

On Kaggle, training is CPU-bound: every epoch re-reads and re-decompresses all 1001 training
cases (``.nii.gz``) and recomputes the same normalization and label regions before any random
augmentation happens. This module runs that fixed part once per session (load, RAS,
resample, crop to the brain, z-score, regions; :func:`train_fixed_part`) and saves the result
as uncompressed ``.npy`` (z-scored image float32, region masks uint8, ~61 MB per BraTS case).
Training then starts from those arrays with the unchanged random part, so the samples are
identical to the uncached pipeline for the same random state; only the speed changes.

Validation and inference do not use the cache.
"""

from __future__ import annotations

import shutil
from collections.abc import Hashable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from monai.transforms import Compose, MapTransform

from btseg.data.dataset import case_to_item
from btseg.data.transforms import (
    DEFAULT_PIXDIM,
    DEFAULT_ROI,
    train_fixed_part,
    train_random_part,
)

BYTES_PER_CASE_ESTIMATE = 64e6  # cropped BraTS case: ~51 MB float32 image + ~10 MB regions
CACHE_FORMAT = "v2"  # bump when the cached content changes


class LoadCachedd(MapTransform):
    """Load cached ``.npy`` arrays as float32 tensors."""

    def __call__(self, data: Mapping[Hashable, object]) -> dict[Hashable, object]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = torch.from_numpy(np.load(d[key]).astype(np.float32, copy=False))
        return d


def get_cached_train_transforms(
    roi_size: Sequence[int] = DEFAULT_ROI, crops_per_case: int = 1
) -> Compose:
    return Compose(
        [LoadCachedd(keys=["image", "label"])] + train_random_part(roi_size, crops_per_case)
    )


def _paths(cache_dir: Path, case_id: str, et_label: int = 4) -> tuple[Path, Path]:
    tag = f"{CACHE_FORMAT}_et{et_label}"
    return cache_dir / f"{case_id}_{tag}_image.npy", cache_dir / f"{case_id}_{tag}_label.npy"


def cache_case(
    case: dict, cache_dir: str | Path, pixdim=DEFAULT_PIXDIM, et_label: int = 4
) -> dict:
    """Run the fixed part for one case and save it; returns the cached item."""
    cache_dir = Path(cache_dir)
    image_path, label_path = _paths(cache_dir, case["id"], et_label)
    if not (image_path.exists() and label_path.exists()):
        d = Compose(train_fixed_part(pixdim, et_label))(case_to_item(case))
        image = np.asarray(d["image"], dtype=np.float32)  # z-scored float32, exact
        tmp_image = image_path.with_suffix(".tmp.npy")
        tmp_label = label_path.with_suffix(".tmp.npy")
        np.save(tmp_image, image)
        np.save(tmp_label, np.asarray(d["label"]).astype(np.uint8))
        tmp_image.replace(image_path)
        tmp_label.replace(label_path)
    return {"id": case["id"], "image": str(image_path), "label": str(label_path)}


def build_cache(
    cases: Sequence[dict],
    cache_dir: str | Path,
    pixdim=DEFAULT_PIXDIM,
    workers: int = 4,
    et_label: int = 4,
) -> list[dict]:
    """Cache all cases (skipping ones already cached). Raises OSError if the disk is too small."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [
        c for c in cases if not all(p.exists() for p in _paths(cache_dir, c["id"], et_label))
    ]
    need = len(missing) * BYTES_PER_CASE_ESTIMATE
    free = shutil.disk_usage(cache_dir).free
    if missing and free < 1.1 * need:
        raise OSError(
            f"Not enough space for the training cache in {cache_dir}: "
            f"{free / 1e9:.1f} GB free, ~{need / 1e9:.1f} GB needed"
        )
    if workers > 1 and len(missing) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            n = len(missing)
            list(pool.map(cache_case, missing, [cache_dir] * n, [tuple(pixdim)] * n,
                          [et_label] * n))
    else:
        for case in missing:
            cache_case(case, cache_dir, pixdim, et_label)
    return [cache_case(c, cache_dir, pixdim, et_label) for c in cases]  # cached: paths only
