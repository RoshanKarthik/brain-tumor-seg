from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from btseg.data.cache import build_cache, get_cached_train_transforms
from btseg.data.dataset import case_to_item
from btseg.data.discover import SEQUENCES
from btseg.data.transforms import get_train_transforms
from conftest import write_case


def make_integer_case(tmp_path):
    """BraTS stores int16 intensities: round the synthetic images and save them as int16."""
    case = write_case(tmp_path / "int", "intcase", shape=(48, 44, 40))
    for s in SEQUENCES:
        img = nib.load(case[s])
        data = np.rint(img.get_fdata() * 10).astype(np.int16)
        nib.save(nib.Nifti1Image(data, img.affine), case[s])
    return case


@pytest.mark.parametrize("kind", ["int16", "float"])
def test_cached_pipeline_gives_identical_samples(tmp_path, kind):
    case = make_integer_case(tmp_path) if kind == "int16" else write_case(
        tmp_path / "f", "floatcase", shape=(48, 44, 40))
    (item,) = build_cache([case], tmp_path / "cache", workers=1)
    stored = np.load(item["image"])
    assert stored.dtype == (np.int16 if kind == "int16" else np.float32)
    assert np.load(item["label"]).dtype == np.uint8

    plain = get_train_transforms(roi_size=(32, 32, 32), crops_per_case=2)
    cached = get_cached_train_transforms(roi_size=(32, 32, 32), crops_per_case=2)
    for seed in (0, 1, 2):
        plain.set_random_state(seed=seed)
        cached.set_random_state(seed=seed)
        a = plain(case_to_item(case))
        b = cached(item)
        assert len(a) == len(b) == 2
        for x, y in zip(a, b, strict=True):
            assert torch.equal(torch.as_tensor(x["image"]), torch.as_tensor(y["image"]))
            assert torch.equal(torch.as_tensor(x["label"]), torch.as_tensor(y["label"]))


def test_build_cache_skips_existing_and_checks_disk(tmp_path, monkeypatch):
    cases = [write_case(tmp_path / f"c{i}", f"c{i}", shape=(24, 24, 24), seed=i) for i in range(2)]
    items = build_cache(cases, tmp_path / "cache", workers=1)
    assert [i["id"] for i in items] == ["c0", "c1"]
    stamp = Path(items[0]["image"]).stat().st_mtime_ns
    build_cache(cases, tmp_path / "cache", workers=1)  # nothing to redo
    assert Path(items[0]["image"]).stat().st_mtime_ns == stamp

    import btseg.data.cache as cache_module

    monkeypatch.setattr(cache_module, "BYTES_PER_CASE_ESTIMATE", 1e18)
    with pytest.raises(OSError, match="Not enough space"):
        build_cache(cases, tmp_path / "other_cache", workers=1)
