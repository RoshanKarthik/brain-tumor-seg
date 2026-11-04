from pathlib import Path

import pytest

from btseg.data.discover import SEQUENCES, discover_cases


def touch_case(directory: Path, case_id: str, parts=(*SEQUENCES, "seg"), ext=".nii.gz"):
    directory.mkdir(parents=True, exist_ok=True)
    for part in parts:
        (directory / f"{case_id}_{part}{ext}").touch()


def test_finds_cases_recursively(tmp_path):
    touch_case(tmp_path / "a" / "BraTS2021_00002", "BraTS2021_00002")
    touch_case(tmp_path / "b" / "deep" / "BraTS2021_00000", "BraTS2021_00000", ext=".nii")
    touch_case(tmp_path / "BraTS2021_00001", "BraTS2021_00001")

    cases = discover_cases(tmp_path, require_seg=True)

    assert [c["id"] for c in cases] == ["BraTS2021_00000", "BraTS2021_00001", "BraTS2021_00002"]
    for case in cases:
        assert set(case) == {"id", *SEQUENCES, "seg"}
        for part in (*SEQUENCES, "seg"):
            path = Path(case[part])
            assert path.is_file()
            assert path.name.startswith(f"{case['id']}_{part}.nii")
    assert cases[0]["flair"].endswith(".nii")


def test_seg_is_optional_by_default(tmp_path):
    touch_case(tmp_path / "c", "case1", parts=SEQUENCES)
    (case,) = discover_cases(tmp_path)
    assert "seg" not in case
    with pytest.raises(FileNotFoundError, match="case1.*missing seg"):
        discover_cases(tmp_path, require_seg=True)


def test_missing_sequence_fails_loudly(tmp_path):
    touch_case(tmp_path / "ok", "ok")
    # has t1ce but no t1: must not be confused
    touch_case(tmp_path / "bad", "bad", parts=("flair", "t1ce", "t2", "seg"))
    with pytest.raises(FileNotFoundError, match=r"bad .*missing t1$"):
        discover_cases(tmp_path)


def test_duplicate_ids_raise(tmp_path):
    touch_case(tmp_path / "x", "dup")
    touch_case(tmp_path / "y", "dup")
    with pytest.raises(ValueError, match="Duplicate"):
        discover_cases(tmp_path)


def test_empty_or_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No"):
        discover_cases(tmp_path)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        discover_cases(tmp_path / "nope")
