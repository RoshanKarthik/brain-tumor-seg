"""MONAI train / val / infer transform pipelines.

Data dicts use ``image`` = list of 4 file paths in the order [FLAIR, T1, T1ce, T2]
(see :func:`btseg.data.dataset.case_to_item`) and, for train/val, ``label`` = the
segmentation file path.

All pipelines: load and stack the 4 sequences channel-first, orient to RAS, resample to
``pixdim`` (1 mm; a no-op for BraTS, needed for user uploads), and z-score each channel
over nonzero (brain) voxels. Labels are converted to [TC, WT, ET] region masks.

Train adds: foreground crop, pad to at least ``roi_size``, random ``roi_size`` crop,
random flips on all 3 axes, and a small random intensity scale/shift.
Val/infer: no cropping; the whole volume goes through sliding-window inference.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

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
    RandSpatialCropd,
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


def _base_transforms(keys: list[str], pixdim: Sequence[float]) -> list:
    has_label = "label" in keys
    return [
        LoadImaged(keys=keys, ensure_channel_first=True, image_only=True),
        EnsureTyped(keys=keys, track_meta=True),
        # explicit labels = NIfTI/nibabel world convention (RAS+); silences MONAI's default change
        Orientationd(keys=keys, axcodes="RAS", labels=(("L", "R"), ("P", "A"), ("I", "S"))),
        Spacingd(
            keys=keys,
            pixdim=tuple(pixdim),
            mode=("bilinear", "nearest") if has_label else "bilinear",
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]


def get_train_transforms(
    roi_size: Sequence[int] = DEFAULT_ROI,
    et_label: int = 4,
    pixdim: Sequence[float] = DEFAULT_PIXDIM,
) -> Compose:
    keys = ["image", "label"]
    return Compose(
        _base_transforms(keys, pixdim)
        + [
            LabelsToRegionsd(keys="label", et_label=et_label),
            CropForegroundd(keys=keys, source_key="image", allow_smaller=True),
            SpatialPadd(keys=keys, spatial_size=tuple(roi_size)),
            RandSpatialCropd(keys=keys, roi_size=tuple(roi_size), random_size=False),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
        ]
    )


def get_val_transforms(
    et_label: int = 4, pixdim: Sequence[float] = DEFAULT_PIXDIM
) -> Compose:
    keys = ["image", "label"]
    return Compose(
        _base_transforms(keys, pixdim) + [LabelsToRegionsd(keys="label", et_label=et_label)]
    )


def get_infer_transforms(pixdim: Sequence[float] = DEFAULT_PIXDIM) -> Compose:
    """Image-only pipeline. Every step is invertible, so predictions can be mapped back."""
    return Compose(_base_transforms(["image"], pixdim))
