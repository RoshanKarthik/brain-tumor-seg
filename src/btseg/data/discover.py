"""Find BraTS cases (FLAIR, T1, T1ce, T2, optional seg) under a root directory.

A case is identified by its FLAIR file, ``<id>_flair.nii[.gz]``, found anywhere below
the root (recursive glob). The other files must sit next to it with the same ``<id>``
prefix. Missing sequences raise an error instead of silently dropping the case.
"""

from __future__ import annotations

from pathlib import Path

SEQUENCES = ("flair", "t1", "t1ce", "t2")
"""Fixed input channel order used everywhere: [FLAIR, T1, T1ce, T2]."""

_EXTENSIONS = (".nii.gz", ".nii")


def _find_file(directory: Path, case_id: str, suffix: str) -> Path | None:
    for ext in _EXTENSIONS:
        path = directory / f"{case_id}_{suffix}{ext}"
        if path.is_file():
            return path
    return None


def discover_cases(root: str | Path, require_seg: bool = False) -> list[dict[str, str]]:
    """Return ``[{id, flair, t1, t1ce, t2, seg?}, ...]`` sorted by id.

    ``seg`` is included only when the segmentation file exists. With ``require_seg=True``
    a missing segmentation is an error. Raises ``FileNotFoundError`` if the root does not
    exist, no case is found, or any case misses a sequence, and ``ValueError`` on
    duplicate case ids.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data root does not exist or is not a directory: {root}")

    flair_files = sorted(
        p for ext in _EXTENSIONS for p in root.rglob(f"*_flair{ext}") if p.is_file()
    )
    if not flair_files:
        raise FileNotFoundError(f"No '*_flair.nii[.gz]' files found under {root}")

    cases: dict[str, dict[str, str]] = {}
    problems: list[str] = []
    for flair in flair_files:
        name = flair.name
        case_id = name[: name.rindex("_flair")]
        if case_id in cases:
            raise ValueError(
                f"Duplicate case id {case_id!r}: {cases[case_id]['flair']} and {flair}"
            )

        case = {"id": case_id}
        missing = []
        for seq in SEQUENCES:
            path = _find_file(flair.parent, case_id, seq)
            if path is None:
                missing.append(seq)
            else:
                case[seq] = str(path)
        seg = _find_file(flair.parent, case_id, "seg")
        if seg is not None:
            case["seg"] = str(seg)
        elif require_seg:
            missing.append("seg")

        if missing:
            problems.append(f"{case_id} ({flair.parent}): missing {', '.join(missing)}")
        cases[case_id] = case

    if problems:
        raise FileNotFoundError(
            f"{len(problems)} case(s) are incomplete:\n  " + "\n  ".join(problems)
        )
    return [cases[k] for k in sorted(cases)]
