"""Locate or extract the BraTS data (e.g. on Kaggle) and return a root for ``discover_cases``.

    python -m btseg.data.prepare --input /kaggle/input --scratch /tmp/brats

1. If extracted cases (``*_flair.nii[.gz]``) already exist under ``--input``, use them.
2. Otherwise extract every ``*.tar`` under ``--input`` into ``--scratch/<tar name>/``
   (keep scratch outside ``/kaggle/working``, which is limited to ~20 GB). Extraction is
   skipped for tars already extracted in this session.
3. A case id found in more than one tar (e.g. a single-case re-upload next to the full
   archive) is kept once: identical copies are dropped silently; differing copies keep
   the version from the smallest tar (the stand-alone re-upload) and print a warning.
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

_FLAIR_GLOBS = ("*_flair.nii.gz", "*_flair.nii")
_DONE_MARKER = ".extracted"


def _flair_files(root: Path) -> list[Path]:
    return sorted(p for g in _FLAIR_GLOBS for p in root.rglob(g) if p.is_file())


def _case_id(flair: Path) -> str:
    return flair.name[: flair.name.rindex("_flair")]


def _case_signature(case_dir: Path, case_id: str) -> list[tuple[str, int]]:
    return sorted((p.name, p.stat().st_size) for p in case_dir.glob(f"{case_id}_*") if p.is_file())


def extract_tars(tars: list[Path], scratch: Path) -> None:
    need = sum(t.stat().st_size for t in tars)
    scratch.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(scratch).free
    pending = [t for t in tars if not (scratch / t.stem / _DONE_MARKER).exists()]
    if pending and free < 1.05 * sum(t.stat().st_size for t in pending):
        raise OSError(
            f"Not enough space in {scratch}: {free / 1e9:.1f} GB free, "
            f"tars total {need / 1e9:.1f} GB"
        )
    for tar_path in pending:
        dest = scratch / tar_path.stem
        dest.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {tar_path} ({tar_path.stat().st_size / 1e9:.2f} GB) -> {dest}")
        with tarfile.open(tar_path) as tar:
            if hasattr(tarfile, "data_filter"):
                tar.extractall(dest, filter="data")  # refuses absolute paths / path traversal
            else:  # pragma: no cover - Python without the 2023 tarfile security backport
                tar.extractall(dest)
        (dest / _DONE_MARKER).touch()


def resolve_duplicates(scratch: Path, tar_sizes: dict[str, int]) -> list[str]:
    """Remove duplicate case folders under ``scratch``; return human-readable notes."""
    by_id: dict[str, list[Path]] = defaultdict(list)
    for flair in _flair_files(scratch):
        by_id[_case_id(flair)].append(flair.parent)

    notes = []
    for case_id, dirs in sorted(by_id.items()):
        if len(dirs) < 2:
            continue
        # top-level folder under scratch = tar it came from; prefer the smallest tar
        dirs.sort(key=lambda d: tar_sizes.get(d.relative_to(scratch).parts[0], 0))
        keep, drop = dirs[0], dirs[1:]
        identical = all(
            _case_signature(d, case_id) == _case_signature(keep, case_id) for d in drop
        )
        for d in drop:
            shutil.rmtree(d)
        if not identical:
            notes.append(
                f"WARNING: {case_id} differs between tars; kept {keep}, removed "
                + ", ".join(str(d) for d in drop)
            )
    return notes


def prepare_data(input_root: str | Path, scratch: str | Path) -> Path:
    input_root, scratch = Path(input_root), Path(scratch)
    if _flair_files(input_root):
        print(f"Found extracted cases under {input_root}")
        return input_root

    tars = sorted(input_root.rglob("*.tar"))
    if not tars:
        raise FileNotFoundError(
            f"No '*_flair.nii[.gz]' files and no '*.tar' archives under {input_root}. "
            "Is the BraTS dataset attached?"
        )
    extract_tars(tars, scratch)
    for note in resolve_duplicates(scratch, {t.stem: t.stat().st_size for t in tars}):
        print(note)
    n = len(_flair_files(scratch))
    print(f"{n} cases under {scratch}")
    return scratch


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Locate or extract the BraTS data.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    args = parser.parse_args(argv)
    root = prepare_data(args.input, args.scratch)
    print(f"DATA_ROOT={root}")
    return root


if __name__ == "__main__":
    main()
