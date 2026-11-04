"""Dataset construction from discovered cases, and the patient-level train/val/test split.

BraTS 2021 has one pre-operative case per patient, so splitting by case id is a
patient-level split.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from pathlib import Path

from monai.data import Dataset

from btseg.data.discover import SEQUENCES

SPLIT_NAMES = ("train", "val", "test")


def case_to_item(case: dict[str, str]) -> dict:
    """Map a discovered case to a MONAI data dict (``image`` in [FLAIR, T1, T1ce, T2] order)."""
    item = {"id": case["id"], "image": [case[seq] for seq in SEQUENCES]}
    if "seg" in case:
        item["label"] = case["seg"]
    return item


def make_splits(
    case_ids: Sequence[str],
    seed: int = 42,
    fractions: Sequence[float] = (0.8, 0.1, 0.1),
) -> dict:
    """Deterministic shuffled split of case ids into train / val / test.

    Val and test get ``round(n * fraction)`` cases each; train gets the rest. The result
    depends only on the set of ids and the seed, not on their input order.
    """
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must be 3 numbers summing to 1, got {fractions}")
    ids = sorted(set(case_ids))
    if len(ids) != len(case_ids):
        raise ValueError("case_ids contains duplicates")
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_val = round(n * fractions[1])
    n_test = round(n * fractions[2])
    n_train = n - n_val - n_test
    return {
        "seed": seed,
        "fractions": list(fractions),
        "n_cases": n,
        "train": sorted(ids[:n_train]),
        "val": sorted(ids[n_train : n_train + n_val]),
        "test": sorted(ids[n_train + n_val :]),
    }


def save_splits(splits: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")


def load_splits(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_split(cases: Sequence[dict], splits: dict, name: str) -> list[dict]:
    """Return the cases of one split, in split order. Fails if any listed id is missing."""
    if name not in SPLIT_NAMES:
        raise ValueError(f"Unknown split {name!r}; expected one of {SPLIT_NAMES}")
    by_id = {c["id"]: c for c in cases}
    missing = [i for i in splits[name] if i not in by_id]
    if missing:
        raise KeyError(
            f"{len(missing)} id(s) of split {name!r} not found in data, e.g. {missing[:5]}"
        )
    return [by_id[i] for i in splits[name]]


def build_dataset(cases: Sequence[dict], transform: Callable) -> Dataset:
    """Plain (non-caching) MONAI Dataset: each case is loaded from disk on access."""
    return Dataset(data=[case_to_item(c) for c in cases], transform=transform)
