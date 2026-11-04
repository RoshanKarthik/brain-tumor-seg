"""Create the fixed-seed patient-level split -> data/splits.json.

Run where the BraTS data is available (e.g. on Kaggle), then commit the JSON:

    python scripts/make_splits.py --data-root /path/to/BraTS2021 --out data/splits.json
"""

from __future__ import annotations

import argparse

from btseg.data.dataset import make_splits, save_splits
from btseg.data.discover import discover_cases


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True, help="Directory searched recursively")
    parser.add_argument("--out", default="data/splits.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fractions",
        type=float,
        nargs=3,
        default=(0.8, 0.1, 0.1),
        metavar=("TRAIN", "VAL", "TEST"),
    )
    args = parser.parse_args(argv)

    cases = discover_cases(args.data_root, require_seg=True)
    splits = make_splits([c["id"] for c in cases], seed=args.seed, fractions=args.fractions)
    save_splits(splits, args.out)
    print(
        f"{splits['n_cases']} cases -> train {len(splits['train'])}, "
        f"val {len(splits['val'])}, test {len(splits['test'])}; wrote {args.out}"
    )
    return splits


if __name__ == "__main__":
    main()
