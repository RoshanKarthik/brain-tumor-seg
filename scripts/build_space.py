"""Assemble app/ + the btseg package into a Space folder and upload it to Hugging Face.

    python scripts/build_space.py --space-id <hf-user>/brain-tumor-seg \\
        --model-repo <hf-user>/brats-segresnet [--dry-run]

Builds ``build/space/`` with app.py, requirements.txt, README.md (model repo filled in),
model_repo.txt (read by app.py at startup) and a copy of ``src/btseg``. Without --dry-run
it creates the Space if needed (Gradio SDK) and uploads the folder; log in first with
``hf auth login``. ZeroGPU itself is chosen afterwards in the Space's Settings.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_FILES = ("app.py", "requirements.txt")


def build(model_repo: str, out: Path) -> Path:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name in APP_FILES:
        shutil.copy2(ROOT / "app" / name, out / name)
    readme = (ROOT / "app" / "README.md").read_text(encoding="utf-8")
    (out / "README.md").write_text(readme.replace("{{MODEL_REPO}}", model_repo), encoding="utf-8")
    (out / "model_repo.txt").write_text(model_repo + "\n", encoding="utf-8")
    shutil.copytree(
        ROOT / "src" / "btseg",
        out / "btseg",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info"),
    )
    return out


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--space-id", required=True, help="<hf-user>/<space-name>")
    p.add_argument("--model-repo", required=True, help="<hf-user>/<model-repo> with best.pt")
    p.add_argument("--out", type=Path, default=ROOT / "build" / "space")
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Build the folder, upload nothing")
    args = p.parse_args(argv)

    folder = build(args.model_repo, args.out)
    files = sorted(f.relative_to(folder).as_posix() for f in folder.rglob("*") if f.is_file())
    print(f"Built {folder} ({len(files)} files): " + ", ".join(
        f for f in files if not f.startswith("btseg/")) + ", btseg/...")
    if args.dry_run:
        print("Dry run: nothing uploaded.")
        return folder

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.space_id, repo_type="space", space_sdk="gradio",
                    private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=args.space_id, repo_type="space",
                      commit_message="Deploy app")
    print(f"Uploaded to https://huggingface.co/spaces/{args.space_id}\n"
          "Next: in the Space's Settings, set the hardware to ZeroGPU.")
    return folder


if __name__ == "__main__":
    main()
