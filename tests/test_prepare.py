import tarfile
from pathlib import Path

import pytest

from btseg.data.discover import SEQUENCES, discover_cases
from btseg.data.prepare import prepare_data


def make_case(directory: Path, case_id: str, payload: bytes = b"x") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for part in (*SEQUENCES, "seg"):
        (directory / f"{case_id}_{part}.nii.gz").write_bytes(payload)


def make_tar(tar_path: Path, src: Path) -> None:
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "w") as tar:
        for child in sorted(src.iterdir()):
            tar.add(child, arcname=child.name)


def test_uses_already_extracted_cases(tmp_path):
    make_case(tmp_path / "input" / "ds" / "BraTS2021_00000", "BraTS2021_00000")
    assert prepare_data(tmp_path / "input", tmp_path / "scratch") == tmp_path / "input"
    assert not (tmp_path / "scratch").exists()


def test_extracts_tars_and_resolves_duplicates(tmp_path, capsys):
    full = tmp_path / "src_full"
    for i in range(3):
        make_case(full / "BraTS2021_Training_Data" / f"BraTS2021_0000{i}", f"BraTS2021_0000{i}")
    # stand-alone re-uploads: 00001 differs from the copy in the full archive, 00002 is identical
    make_case(tmp_path / "src_1" / "BraTS2021_00001", "BraTS2021_00001", payload=b"fixed!")
    make_case(tmp_path / "src_2" / "BraTS2021_00002", "BraTS2021_00002")
    inp = tmp_path / "input" / "brats-2021-task1"
    make_tar(inp / "BraTS2021_Training_Data.tar", full)
    make_tar(inp / "BraTS2021_00001.tar", tmp_path / "src_1")
    make_tar(inp / "BraTS2021_00002.tar", tmp_path / "src_2")

    root = prepare_data(tmp_path / "input", tmp_path / "scratch")
    cases = discover_cases(root, require_seg=True)

    assert [c["id"] for c in cases] == ["BraTS2021_00000", "BraTS2021_00001", "BraTS2021_00002"]
    fixed = next(c for c in cases if c["id"] == "BraTS2021_00001")
    assert Path(fixed["flair"]).read_bytes() == b"fixed!"
    out = capsys.readouterr().out
    assert "WARNING: BraTS2021_00001 differs" in out
    assert "BraTS2021_00002 differs" not in out

    # second call in the same session does not extract again
    prepare_data(tmp_path / "input", tmp_path / "scratch")
    assert "Extracting" not in capsys.readouterr().out


def test_no_data_fails_loudly(tmp_path):
    (tmp_path / "input").mkdir()
    with pytest.raises(FileNotFoundError, match="dataset attached"):
        prepare_data(tmp_path / "input", tmp_path / "scratch")
