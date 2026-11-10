"""MONAI train / val / infer transform pipelines.

Data dicts use ``image`` = list of 4 file paths in the order [FLAIR, T1, T1ce, T2]
(see :func:`btseg.data.dataset.case_to_item`) and, for train/val, ``label`` = the
segmentation file path.

All pipelines: load and stack the 4 sequences channel-first, orient to RAS, resample to
``pixdim`` (1 mm; a no-op for BraTS, needed for user uploads), and z-score each channel
over nonzero (brain) voxels. Labels are converted to [TC, WT, ET] region masks.

Train: crop to the brain (nonzero voxels of the *raw* image), then normalize, convert
labels, pad to at least ``roi_size``, take ``crops_per_case`` random ``roi_size`` crops,
random flips on all 3 axes, and a small random intensity scale/shift.
Val/infer: no cropping; the whole volume goes through sliding-window inference.

Speed (training is CPU-bound on Kaggle): train/val keep the stored dtype (int16 for
BraTS) until the explicit float32 cast, and train does all per-voxel work after the
brain crop. Several crops per loaded case amortize the cost of reading the files.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

import numpy as np
import torch
from monai.config import KeysCollection
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropSamplesd,
    Spacingd,
    SpatialPadd,
)

from btseg.data.labels import labels_to_regions

DEFAULT_ROI = (128, 128, 128)
DEFAULT_PIXDIM = (1.0, 1.0, 1.0)


class LabelsToRegionsd(MapTransform):
    """Replace a (1, H, W, D) BraTS label map with float (3, H, W, D) [TC, WT, ET] masks."""

    def __init__(self, keys: KeysCollection, et_label: int = 4, allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.et_label = et_label

    def __call__(self, data: Mapping[Hashable, object]) -> dict[Hashable, object]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = labels_to_regions(d[key], et_label=self.et_label).float()
        return d


def _load_and_resample(keys: list[str], pixdim: Sequence[float], keep_dtype: bool) -> list:
    """Load, orient to RAS, resample. ``keep_dtype`` defers the float32 conversion."""
    has_label = "label" in keys
    return [
        LoadImaged(
            keys=keys,
            ensure_channel_first=True,
            image_only=True,
            dtype=None if keep_dtype else np.float32,
        ),
        EnsureTyped(keys=keys, track_meta=True),
        # explicit labels = NIfTI/nibabel world convention (RAS+); silences MONAI's default change
        Orientationd(keys=keys, axcodes="RAS", labels=(("L", "R"), ("P", "A"), ("I", "S"))),
        Spacingd(
            keys=keys,
            pixdim=tuple(pixdim),
            mode=("bilinear", "nearest") if has_label else "bilinear",
        ),
    ]


def _normalize() -> list:
    return [
        EnsureTyped(keys="image", dtype=torch.float32, track_meta=True),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]


def train_fixed_part(pixdim: Sequence[float] = DEFAULT_PIXDIM) -> list:
    """Deterministic start of the train pipeline: load, RAS, resample, crop to the brain.

    Its output can be cached on disk once (:mod:`btseg.data.cache`).
    """
    keys = ["image", "label"]
    return _load_and_resample(keys, pixdim, keep_dtype=True) + [
        CropForegroundd(keys=keys, source_key="image", allow_smaller=True)
    ]


def train_random_part(
    roi_size: Sequence[int] = DEFAULT_ROI, et_label: int = 4, crops_per_case: int = 1
) -> list:
    """Per-epoch part of the train pipeline: normalize, regions, pad, random crops, augment."""
    keys = ["image", "label"]
    return _normalize() + [
        LabelsToRegionsd(keys="label", et_label=et_label),
        SpatialPadd(keys=keys, spatial_size=tuple(roi_size)),
        RandSpatialCropSamplesd(
            keys=keys, roi_size=tuple(roi_size), num_samples=crops_per_case, random_size=False
        ),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ]


def get_train_transforms(
    roi_size: Sequence[int] = DEFAULT_ROI,
    et_label: int = 4,
    pixdim: Sequence[float] = DEFAULT_PIXDIM,
    crops_per_case: int = 1,
) -> Compose:
    """Returns a list of ``crops_per_case`` dicts per case (MONAI's collate flattens it)."""
    return Compose(
        train_fixed_part(pixdim) + train_random_part(roi_size, et_label, crops_per_case)
    )


def get_val_transforms(
    et_label: int = 4, pixdim: Sequence[float] = DEFAULT_PIXDIM
) -> Compose:
    keys = ["image", "label"]
    return Compose(
        _load_and_resample(keys, pixdim, keep_dtype=True)
        + [LabelsToRegionsd(keys="label", et_label=et_label)]
        + _normalize()
    )


def get_infer_transforms(pixdim: Sequence[float] = DEFAULT_PIXDIM) -> Compose:
    """Image-only pipeline. Every step is invertible, so predictions can be mapped back."""
    return Compose(_load_and_resample(["image"], pixdim, keep_dtype=False) + _normalize())
