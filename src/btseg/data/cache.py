"""Disk cache of the deterministic part of the training pipeline (speed only).

On Kaggle, training is CPU-bound: every epoch re-reads and re-decompresses all 1001 training
cases (``.nii.gz``) before any random augmentation happens. This module runs the fixed part
once per session (load, RAS, resample, crop to the brain; :func:`train_fixed_part`) and saves
the result as uncompressed ``.npy`` (image int16 when the data is integer, label uint8).
Training then starts from those arrays with the unchanged random part, so the samples are
identical to the uncached pipeline for the same random state; only the I/O is faster.

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

BYTES_PER_CASE_ESTIMATE = 32e6  # cropped BraTS case: ~25 MB int16 image + ~3 MB label


class LoadCachedd(MapTransform):
    """Load cached ``.npy`` arrays as float32 tensors (exact for the cached int16 data)."""

    def __call__(self, data: Mapping[Hashable, object]) -> dict[Hashable, object]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = torch.from_numpy(np.load(d[key]).astype(np.float32))
        return d


def get_cached_train_transforms(
    roi_size: Sequence[int] = DEFAULT_ROI, et_label: int = 4, crops_per_case: int = 1
) -> Compose:
    return Compose(
        [LoadCachedd(keys=["image", "label"])]
        + train_random_part(roi_size, et_label, crops_per_case)
    )


def _paths(cache_dir: Path, case_id: str) -> tuple[Path, Path]:
    return cache_dir / f"{case_id}_image.npy", cache_dir / f"{case_id}_label.npy"


def cache_case(case: dict, cache_dir: str | Path, pixdim=DEFAULT_PIXDIM) -> dict:
    """Run the fixed part for one case and save it; returns the cached item."""
    cache_dir = Path(cache_dir)
    image_path, label_path = _paths(cache_dir, case["id"])
    if not (image_path.exists() and label_path.exists()):
        d = Compose(train_fixed_part(pixdim))(case_to_item(case))
        image = np.asarray(d["image"])
        info = np.iinfo(np.int16)
        fits = info.min <= image.min() and image.max() <= info.max
        if fits and np.array_equal(image, np.round(image)):
            image = image.astype(np.int16)  # lossless for integer MRI data
        else:
            image = image.astype(np.float32)
        tmp_image = image_path.with_suffix(".tmp.npy")
        tmp_label = label_path.with_suffix(".tmp.npy")
        np.save(tmp_image, image)
        np.save(tmp_label, np.asarray(d["label"]).astype(np.uint8))
        tmp_image.replace(image_path)
        tmp_label.replace(label_path)
    return {"id": case["id"], "image": str(image_path), "label": str(label_path)}


def build_cache(
    cases: Sequence[dict], cache_dir: str | Path, pixdim=DEFAULT_PIXDIM, workers: int = 4
) -> list[dict]:
    """Cache all cases (skipping ones already cached). Raises OSError if the disk is too small."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [c for c in cases if not all(p.exists() for p in _paths(cache_dir, c["id"]))]
    need = len(missing) * BYTES_PER_CASE_ESTIMATE
    free = shutil.disk_usage(cache_dir).free
    if missing and free < 1.1 * need:
        raise OSError(
            f"Not enough space for the training cache in {cache_dir}: "
            f"{free / 1e9:.1f} GB free, ~{need / 1e9:.1f} GB needed"
        )
    if workers > 1 and len(missing) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(pool.map(cache_case, missing, [cache_dir] * len(missing),
                          [tuple(pixdim)] * len(missing)))
    else:
        for case in missing:
            cache_case(case, cache_dir, pixdim)
    return [cache_case(c, cache_dir, pixdim) for c in cases]  # all cached now: just paths
