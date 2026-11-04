"""Build a Kaggle kernel folder from notebooks/kaggle_train.ipynb, ready for `kaggle kernels push`.

    python scripts/kaggle_push.py --user <kaggle-username> --mode smoke
    kaggle kernels push -p build/kaggle/brats-segresnet-smoke

The notebook copy gets its settings cell rewritten (smoke vs. full run, config, branch);
the metadata attaches the BraTS 2021 dataset, a T4 GPU and internet. Kernels are private.
``--resume`` (full runs) also attaches the kernel's own previous version output, where the
notebook finds ``runs/<run_name>/last.pt``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_train.ipynb"
DATASET = "dschettler8845/brats-2021-task1"


def _set(src: str, name: str, value: str) -> str:
    pattern = rf"^{name} = .*?(\s+#.*)?$"
    if not re.search(pattern, src, flags=re.M):
        raise ValueError(f"Setting {name} not found in the notebook settings cell")

    def repl(m: re.Match) -> str:
        return f"{name} = {value}{m.group(1) or ''}"

    return re.sub(pattern, repl, src, count=1, flags=re.M)


def build(
    user: str,
    mode: str,
    config: str,
    branch: str,
    out_root: Path,
    slug: str | None = None,
    resume: bool = False,
) -> Path:
    stem = Path(config).stem
    slug = slug or (f"brats-{stem}-smoke" if mode == "smoke" else f"brats-{stem}-train")
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    settings = next(c for c in nb["cells"] if "# ---- settings ----" in "".join(c["source"]))
    src = "".join(settings["source"])
    src = _set(src, "BRANCH", json.dumps(branch))
    src = _set(src, "CONFIG", json.dumps(config))
    src = _set(src, "RUN_SMOKE_TEST", str(mode == "smoke"))
    src = _set(src, "RUN_FULL_TRAINING", str(mode == "full"))
    lines = src.split("\n")
    settings["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]

    folder = out_root / slug
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "kaggle_train.ipynb").write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    metadata = {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": "kaggle_train.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [DATASET],
        "competition_sources": [],
        "kernel_sources": [f"{user}/{slug}"] if resume else [],
        "model_sources": [],
    }
    meta_text = json.dumps(metadata, indent=2) + "\n"
    (folder / "kernel-metadata.json").write_text(meta_text, encoding="utf-8")
    return folder


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--user", required=True, help="Kaggle username")
    p.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    p.add_argument("--config", default="configs/segresnet.yaml")
    p.add_argument("--branch", default="main")
    p.add_argument("--slug", help="Kernel slug; default brats-<config>-smoke / -train")
    p.add_argument(
        "--resume", action="store_true", help="Attach the previous version's output (full runs)"
    )
    p.add_argument("--out", type=Path, default=ROOT / "build" / "kaggle")
    args = p.parse_args(argv)
    if args.resume and args.mode != "full":
        p.error("--resume only applies to --mode full")
    folder = build(
        args.user, args.mode, args.config, args.branch, args.out, args.slug, args.resume
    )
    print(f"Wrote {folder}\nNext: kaggle kernels push -p {folder}")
    return folder


if __name__ == "__main__":
    main()
