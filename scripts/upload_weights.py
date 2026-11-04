"""Push a trained model + model card to a Hugging Face model repo.

    python scripts/upload_weights.py --repo-id <hf-user>/brats-segresnet \\
        --run-dir runs/segresnet_dice --results results/segresnet_dice --dry-run

Stages ``best.pt``, ``config.json``, ``train_log.csv``, optionally ``model.onnx`` +
``export_info.json`` (from --run-dir or --onnx) and the evaluation outputs (summary.json,
per_case.csv, figures) under ``results/``, plus a generated ``README.md`` model card, in
``build/hf_upload/<repo-name>/``. Without --dry-run it then uploads that folder (log in
first with ``hf auth login``, or set HF_TOKEN). Every number in the card is read from
these files; missing results show as TBD.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REGIONS = ("WT", "TC", "ET")
VOLUME_REGIONS = ("NCR", "ED", "ET", "TC", "WT")
MODEL_NAMES = {"segresnet": "SegResNet", "unet3d": "3D U-Net", "swinunetr": "SwinUNETR"}
RESULT_FILES = ("summary.json", "per_case.csv", "ablation.csv")


def _fmt(value, digits=3) -> str:
    return "TBD" if value is None else f"{value:.{digits}f}"


def model_card(
    repo_id: str,
    checkpoint: dict,
    summary: dict | None,
    splits: dict | None,
    export_info: dict | None,
) -> str:
    cfg = checkpoint["config"]
    model, data, train = cfg["model"], cfg["data"], cfg["train"]
    name = MODEL_NAMES.get(model["name"], model["name"])
    n_params = sum(t.numel() for t in checkpoint["model"].values() if t.is_floating_point())
    kwargs = ", ".join(f"{k}={v}" for k, v in model["kwargs"].items()) or "defaults"
    split_line = (
        f"{len(splits['train'])} train / {len(splits['val'])} validation / "
        f"{len(splits['test'])} test cases (patient-level, seed {splits.get('seed', 42)})"
        if splits else "TBD"
    )
    val = checkpoint.get("val_dice") or {}
    best = (
        f"epoch {checkpoint['epoch']}, mean validation Dice {_fmt(checkpoint.get('val_dice_mean'))}"
        f" (TC {_fmt(val.get('TC'))}, WT {_fmt(val.get('WT'))}, ET {_fmt(val.get('ET'))})"
        if "epoch" in checkpoint else "TBD"
    )

    lines = [
        "---",
        "tags: [medical-imaging, image-segmentation, 3d, brain-tumor, brats, mri, monai]",
        "pipeline_tag: image-segmentation",
        "---",
        "",
        f"# Brain tumor segmentation: {name} on BraTS 2021 MRI",
        "",
        "> **Research and education only. Not a medical device and not for clinical use.**",
        "",
        "Segments brain tumors in multi-sequence MRI into necrotic core (NCR), peritumoral "
        "edema (ED) and enhancing tumor (ET), and is used to compute their volumes. "
        "Code: <https://github.com/RoshanKarthik/brain-tumor-seg>.",
        "",
        "## Model",
        "",
        f"- Architecture: MONAI {name} ({kwargs}), {n_params:,} parameters",
        "- Input: 4 co-registered MRI sequences in the order [FLAIR, T1, T1ce, T2], reoriented "
        f"to RAS, resampled to {data['pixdim']} mm, z-scored per channel over brain voxels",
        "- Output: 3 sigmoid channels [TC, WT, ET] (tumor core, whole tumor, enhancing tumor), "
        f"thresholded at {cfg.get('postprocess', {}).get('threshold', 0.5)} and converted to "
        f"BraTS labels 1 = NCR, 2 = ED, {data['et_label']} = ET",
        f"- Inference: sliding window, ROI {cfg['infer']['roi_size']}, overlap "
        f"{cfg['infer']['overlap']}, {cfg['infer']['mode']} blending; optional 8-flip TTA",
        "",
        "## Training",
        "",
        "- Data: BraTS 2021 Task 1 training set (adult glioma, pre-operative, "
        "skull-stripped, 1 mm), " + split_line,
        f"- Loss: {cfg['loss']['name']} (sigmoid), optimizer AdamW (lr {train['lr']}, "
        f"weight decay {train['weight_decay']}), {train['warmup_epochs']} warmup epochs then "
        f"cosine decay, {train['epochs']} epochs, mixed precision",
        f"- Batches: {data['batch_size']} case(s) x {data.get('crops_per_case', 1)} random "
        f"{data['roi_size']} crop(s) per step; random flips and intensity scale/shift",
        f"- Best checkpoint: {best}",
        "",
    ]

    lines += ["## Results", ""]
    if summary:
        lines += [
            f"Evaluated on the held-out **{summary.get('split', 'test')}** split "
            f"({summary['n_cases']} cases), TTA {'on' if summary['tta'] else 'off'}, "
            f"post-processing {summary['postprocess']}. {summary['metric_note']}",
            "",
            "| Region | Dice mean ± std | Dice median | HD95 mean (mm) | HD95 median (mm) |",
            "|---|---|---|---|---|",
        ]
        for r in REGIONS:
            d, h = summary["dice"][r], summary["hd95_mm"][r]
            lines.append(
                f"| {r} | {_fmt(d['mean'])} ± {_fmt(d['std'])} | {_fmt(d['median'])} | "
                f"{_fmt(h['mean'], 2)} | {_fmt(h['median'], 2)} |"
            )
        lines += [
            "",
            "Volume agreement (predicted vs. ground truth, cm³):",
            "",
            "| Region | MAE (cm³) | Mean abs. % error | Pearson r | Bias (cm³) |",
            "|---|---|---|---|---|",
        ]
        for r in VOLUME_REGIONS:
            v = summary["volume_agreement"][r]
            lines.append(
                f"| {r} | {_fmt(v['mae_cm3'], 2)} | {_fmt(v['mean_abs_percent_error'], 1)} "
                f"(n = {v['percent_error_n']}) | {_fmt(v['pearson_r'])} | "
                f"{_fmt(v['bias_cm3'], 2)} |"
            )
        lines += ["", "Per-case results and figures are in `results/`.", ""]
    else:
        lines += ["TBD: not evaluated on the test split yet.", ""]

    if export_info:
        lines += [
            "## ONNX",
            "",
            f"`model.onnx`: opset {export_info['opset']}, fixed {export_info['roi_size']} patch, "
            f"dynamic batch; max |ONNX − PyTorch| = {export_info['parity_max_abs_diff']:.1e} on "
            "a random input. Use with sliding-window inference (see `btseg.export`).",
            "",
        ]

    lines += [
        "## Usage",
        "",
        "```python",
        "from btseg.infer import Predictor  # pip install git+https://github.com/RoshanKarthik/brain-tumor-seg",
        "",
        f'predictor = Predictor.from_checkpoint("{repo_id}")',
        'result = predictor.predict("flair.nii.gz", "t1.nii.gz", "t1ce.nii.gz", "t2.nii.gz")',
        "print(result.volumes)  # NCR, ED, ET, TC, WT in cm³",
        "```",
        "",
        "## Intended use and limitations",
        "",
        "- Intended for research and teaching on brain tumor segmentation and volumetry.",
        "- **Not** for diagnosis, treatment planning or any clinical decision.",
        "- MRI only: trained on FLAIR, T1, T1ce and T2. CT is not supported.",
        "- Adult gliomas, pre-operative scans, BraTS preprocessing (co-registered, "
        "skull-stripped). Other tumor types, post-operative scans or unprocessed images may "
        "fail silently.",
        "- No external validation: all numbers come from a held-out part of BraTS 2021.",
        "",
        "## Data license and citation",
        "",
        "Trained on the BraTS 2021 Task 1 data, which is subject to the BraTS / Synapse data "
        "terms; no scans are redistributed here. If you use this model, cite:",
        "",
        "- U. Baid et al., *The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor "
        "Segmentation and Radiogenomic Classification*, arXiv:2107.02314, 2021.",
        "- B. H. Menze et al., *The Multimodal Brain Tumor Image Segmentation Benchmark "
        "(BRATS)*, IEEE TMI 34(10), 2015.",
        "- S. Bakas et al., *Advancing The Cancer Genome Atlas glioma MRI collections with "
        "expert segmentation labels and radiomic features*, Scientific Data 4, 2017.",
        "- A. Myronenko, *3D MRI brain tumor segmentation using autoencoder regularization*, "
        "BrainLes 2018 (SegResNet); MONAI Consortium, *MONAI*, 2020.",
        "",
    ]
    return "\n".join(lines)


def stage(args) -> Path:
    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "best.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"{ckpt_path} not found")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    out = Path(args.staging) / args.repo_id.split("/")[-1]
    if out.exists():
        shutil.rmtree(out)
    (out / "results").mkdir(parents=True)
    shutil.copy2(ckpt_path, out / "best.pt")
    for name in ("config.json", "train_log.csv"):
        if (run_dir / name).is_file():
            shutil.copy2(run_dir / name, out / name)

    export_info = None
    onnx = Path(args.onnx) if args.onnx else run_dir / "model.onnx"
    if onnx.is_file():
        shutil.copy2(onnx, out / "model.onnx")
        info_path = onnx.with_name("export_info.json")
        if info_path.is_file():
            shutil.copy2(info_path, out / "export_info.json")
            export_info = json.loads(info_path.read_text(encoding="utf-8"))

    summary = None
    if args.results:
        results = Path(args.results)
        for f in [*(results / n for n in RESULT_FILES), *sorted(results.glob("*.png"))]:
            if f.is_file():
                shutil.copy2(f, out / "results" / f.name)
        if (results / "summary.json").is_file():
            summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))

    splits_path = Path(args.splits)
    splits = json.loads(splits_path.read_text(encoding="utf-8")) if splits_path.is_file() else None
    card = model_card(args.repo_id, checkpoint, summary, splits, export_info)
    (out / "README.md").write_text(card, encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description="Upload weights + model card to the HF Hub.")
    p.add_argument("--repo-id", required=True, help="<hf-user>/<model-repo>")
    p.add_argument("--run-dir", required=True, help="Training output with best.pt")
    p.add_argument("--results", help="Evaluation output dir (summary.json, figures)")
    p.add_argument("--onnx", help="model.onnx (default: <run-dir>/model.onnx if present)")
    p.add_argument("--splits", default=str(ROOT / "data" / "splits.json"))
    p.add_argument("--staging", default=str(ROOT / "build" / "hf_upload"))
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Stage files and card, upload nothing")
    args = p.parse_args(argv)

    folder = stage(args)
    files = sorted(f for f in folder.rglob("*") if f.is_file())
    print(f"Staged {len(files)} files in {folder}:")
    for f in files:
        print(f"  {f.relative_to(folder).as_posix():32s} {f.stat().st_size / 1e6:8.2f} MB")
    if args.dry_run:
        print("Dry run: nothing uploaded. Review README.md, then rerun without --dry-run.")
        return folder

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=args.repo_id, repo_type="model",
                      commit_message="Upload model weights and card")
    print(f"Uploaded to https://huggingface.co/{args.repo_id}")
    return folder


if __name__ == "__main__":
    main()
