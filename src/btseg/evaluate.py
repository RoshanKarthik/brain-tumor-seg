"""Evaluate a checkpoint on a split -> results/<run_name>/ (CSV, JSON, figures).

    python -m btseg.evaluate --ckpt runs/segresnet_dice/best.pt --data-root /data/brats \\
        --split test [--tta] [--ablation]

Writes to ``<out-root>/<run_name>`` (``<run_name>_<split>`` for splits other than test):
- ``per_case.csv``: Dice and HD95 for WT/TC/ET, predicted and ground-truth volume (cm^3)
  of NCR, ED, ET, TC, WT for every case
- ``summary.json``: mean, std, median per metric; volume agreement (MAE, mean absolute %
  error over cases with a nonzero ground-truth volume, Pearson r, Bland-Altman bias and
  limits of agreement); settings and provenance
- ``dice_boxplot.png``, ``bland_altman.png``, ``volume_scatter.png``, ``overlays.png``
  (best / median / worst cases by mean Dice)
- ``ablation.csv`` with ``--ablation``: TTA off/on x post-processing off/on, computed
  from the same forward passes (inference-only, no retraining)
- ``predictions/<case>_seg.nii.gz`` for the main setting (not committed; *.nii.gz ignored)
- ``pp_grid.csv`` and ``threshold_grid.csv`` with ``--pp-grid``: mean Dice for a grid of
  post-processing settings (small-component removal x small-ET -> NCR) and for per-region
  thresholds, computed from the same predictions. Use them on the val split to choose
  ``--region-thresholds``, ``--min-component-voxels`` and ``--et-min-voxels`` for test.

Metric conventions for empty masks follow BraTS, see :mod:`btseg.metrics`. The test
split is meant to be evaluated once, for the final numbers; tune post-processing on val.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import subprocess
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import stats

import btseg
from btseg.config import PostprocessConfig
from btseg.data.dataset import load_splits, select_split
from btseg.data.discover import SEQUENCES, discover_cases
from btseg.data.labels import labels_to_regions
from btseg.infer import Predictor, save_label_map
from btseg.metrics import BRATS_HD95_ONE_EMPTY, EVAL_REGIONS, dice, region_metrics
from btseg.postprocess import postprocess
from btseg.viz import (
    plot_bland_altman,
    plot_case_overlays,
    plot_dice_boxplot,
    plot_volume_scatter,
)
from btseg.volumetrics import VOLUME_REGIONS, region_volumes

METRIC_NOTE = (
    "Empty masks follow BraTS: Dice 1 / HD95 0 mm if prediction and ground truth are both "
    f"empty, Dice 0 / HD95 {BRATS_HD95_ONE_EMPTY} mm if exactly one is empty. HD95 is MONAI's "
    "symmetric 95th-percentile Hausdorff distance in mm."
)


MIN_COMPONENT_GRID = (0, 50, 200, 500)
ET_MIN_GRID = (0, 100, 300, 500, 800)
THRESHOLD_GRID = (0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7)
PROB_REGIONS = ("TC", "WT", "ET")  # channel order of the probabilities / region masks


def _num(x) -> float | None:
    x = float(x)
    return None if np.isnan(x) else x


def describe(values) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {
        "mean": _num(v.mean()),
        "std": _num(v.std(ddof=1)) if len(v) > 1 else None,
        "median": _num(np.median(v)),
        "n": int(len(v)),
    }


def volume_agreement(pred, gt) -> dict:
    p, g = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
    diff = p - g
    nonzero = g > 0
    sd = diff.std(ddof=1) if len(diff) > 1 else float("nan")
    r = None
    if len(p) > 1 and p.std() > 0 and g.std() > 0:
        r = _num(stats.pearsonr(p, g)[0])
    return {
        "mae_cm3": _num(np.abs(diff).mean()),
        "mean_abs_percent_error": _num(100 * np.mean(np.abs(diff[nonzero]) / g[nonzero]))
        if nonzero.any()
        else None,
        "percent_error_n": int(nonzero.sum()),
        "pearson_r": r,
        "bias_cm3": _num(diff.mean()),
        "loa_cm3": [_num(diff.mean() - 1.96 * sd), _num(diff.mean() + 1.96 * sd)],
    }


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, cwd=Path(__file__).parent,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pick_overlay_cases(mean_dice: list[float], k: int = 2) -> list[tuple[str, int]]:
    """(label, index) for the k best, k closest to the median and k worst, no repeats."""
    order = list(np.argsort(mean_dice))  # worst first
    median = float(np.median(mean_dice))
    by_median = sorted(range(len(mean_dice)), key=lambda i: abs(mean_dice[i] - median))
    picked, seen = [], set()
    for label, candidates in (("best", order[::-1]), ("median", by_median), ("worst", order)):
        taken = 0
        for i in candidates:
            if taken == k:
                break
            if int(i) not in seen:
                picked.append((label, int(i)))
                seen.add(int(i))
                taken += 1
    return picked


def grid_dice(probabilities, gt_regions, base_pp: PostprocessConfig) -> dict:
    """Dice [WT, TC, ET] for every (min_component_voxels, et_min_voxels) in the grid,
    using the thresholds of ``base_pp``."""
    out = {}
    for min_component in MIN_COMPONENT_GRID:
        base = postprocess(
            probabilities,
            replace(base_pp, min_component_voxels=min_component, et_min_voxels=0),
        )
        for et_min in ET_MIN_GRID:
            regions = base.copy()  # same small-ET rule as btseg.postprocess
            if et_min > 0 and 0 < regions[2].sum() < et_min:
                regions[2] = False
            out[(min_component, et_min)] = [
                dice(regions[i], gt_regions[i]) for i in (1, 0, 2)  # WT, TC, ET
            ]
    return out


def threshold_dice(probabilities, gt_regions) -> dict[tuple[str, float], float]:
    """Dice of each region when only that region's threshold is changed (others at 0.5)."""
    out = {}
    for i, region in enumerate(PROB_REGIONS):
        for t in THRESHOLD_GRID:
            thresholds = [0.5, 0.5, 0.5]
            thresholds[i] = t
            regions = postprocess(probabilities,
                                  PostprocessConfig(region_thresholds=tuple(thresholds)))
            out[(region, t)] = dice(regions[i], gt_regions[i])
    return out


def evaluate(
    predictor: Predictor,
    cases: list[dict],
    out_dir: Path,
    tta: bool = False,
    ablation: bool = False,
    pp_variant: PostprocessConfig | None = None,
    provenance: dict | None = None,
    pp_grid: bool = False,
) -> dict:
    cfg = predictor.config
    et = cfg.data.et_label
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    # "main" = the configured setting; the ablation adds TTA off/on x post-processing off/on
    main_name = "main"
    variants = {main_name: (tta, cfg.postprocess)}
    if ablation:
        threshold_only = PostprocessConfig()  # plain 0.5 threshold, nothing tuned
        full = pp_variant or cfg.postprocess
        for use_tta in (False, True):
            prefix = "tta" if use_tta else "no_tta"
            variants[f"{prefix}+threshold_only"] = (use_tta, threshold_only)
            variants[f"{prefix}+postprocess"] = (use_tta, full)
    rows: dict[str, list[dict]] = {name: [] for name in variants}
    grid: dict[tuple[int, int], list] = {}
    thr_grid: dict[tuple[str, float], list] = {}
    seconds = []

    for n, case in enumerate(cases, start=1):
        t0 = time.perf_counter()
        maps = predictor.predict_maps([case[s] for s in SEQUENCES], tta=tta or ablation)
        seconds.append(time.perf_counter() - t0)
        gt = np.rint(nib.load(case["seg"]).get_fdata()).astype(np.uint8)
        gt_vol = region_volumes(gt, maps.affine, et)
        if pp_grid:
            probs = maps.tta_mean if tta else maps.plain
            gt_regions = labels_to_regions(gt, et)
            for key, d in grid_dice(probs, gt_regions, cfg.postprocess).items():
                grid.setdefault(key, []).append(d)
            for key, d in threshold_dice(probs, gt_regions).items():
                thr_grid.setdefault(key, []).append(d)
        for name, (use_tta, pp) in variants.items():
            labels = predictor.labels_from(maps.tta_mean if use_tta else maps.plain, pp)
            if name == main_name:
                save_label_map(labels, case["flair"], pred_dir / f"{case['id']}_seg.nii.gz")
            m = region_metrics(labels, gt, maps.affine, et)
            pred_vol = region_volumes(labels, maps.affine, et)
            row = {"case_id": case["id"]}
            row.update({f"dice_{r}": round(m[r]["dice"], 6) for r in EVAL_REGIONS})
            row.update({f"hd95_{r}": round(m[r]["hd95"], 4) for r in EVAL_REGIONS})
            for r in VOLUME_REGIONS:
                row[f"vol_pred_{r}"] = round(pred_vol[r]["cm3"], 4)
                row[f"vol_gt_{r}"] = round(gt_vol[r]["cm3"], 4)
            rows[name].append(row)
        main_row = rows[main_name][-1]
        print(
            f"[{n}/{len(cases)}] {case['id']} "
            + " ".join(f"{r} {main_row[f'dice_{r}']:.3f}" for r in EVAL_REGIONS)
            + f" ({seconds[-1]:.1f}s)",
            flush=True,
        )

    main_rows = rows[main_name]
    _write_csv(out_dir / "per_case.csv", main_rows)
    col = {k: np.array([r[k] for r in main_rows]) for k in main_rows[0] if k != "case_id"}

    summary = {
        **(provenance or {}),
        "n_cases": len(main_rows),
        "tta": tta,
        "postprocess": dataclasses.asdict(cfg.postprocess),
        "device": str(predictor.device),
        "mean_seconds_per_case": round(float(np.mean(seconds)), 3),
        "metric_note": METRIC_NOTE,
        "dice": {r: describe(col[f"dice_{r}"]) for r in EVAL_REGIONS},
        "hd95_mm": {r: describe(col[f"hd95_{r}"]) for r in EVAL_REGIONS},
        "volume_agreement": {
            r: volume_agreement(col[f"vol_pred_{r}"], col[f"vol_gt_{r}"]) for r in VOLUME_REGIONS
        },
    }
    summary["dice"]["mean_of_regions"] = _num(
        np.mean([summary["dice"][r]["mean"] for r in EVAL_REGIONS])
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if ablation:
        table = []
        for name, (use_tta, pp) in variants.items():
            if name == main_name:
                continue
            c = {k: np.array([r[k] for r in rows[name]]) for k in rows[name][0] if k != "case_id"}
            entry = {"variant": name, "tta": use_tta, **{
                f"pp_{k}": v for k, v in dataclasses.asdict(pp).items()
            }}
            for r in EVAL_REGIONS:
                entry[f"dice_{r}_mean"] = round(float(c[f"dice_{r}"].mean()), 4)
            entry["dice_mean_of_regions"] = round(
                float(np.mean([entry[f"dice_{r}_mean"] for r in EVAL_REGIONS])), 4
            )
            for r in EVAL_REGIONS:
                entry[f"hd95_{r}_mean"] = round(float(c[f"hd95_{r}"].mean()), 3)
            table.append(entry)
        _write_csv(out_dir / "ablation.csv", table)

    if pp_grid:
        table = []
        for (min_component, et_min), values in grid.items():
            v = np.asarray(values)
            entry = {"min_component_voxels": min_component, "et_min_voxels": et_min, "tta": tta}
            entry.update({f"dice_{r}_mean": round(float(v[:, i].mean()), 4)
                          for i, r in enumerate(EVAL_REGIONS)})
            entry["dice_mean_of_regions"] = round(float(v.mean()), 4)
            table.append(entry)
        table.sort(key=lambda e: -e["dice_mean_of_regions"])
        _write_csv(out_dir / "pp_grid.csv", table)
        _write_csv(out_dir / "threshold_grid.csv", [
            {"region": region, "threshold": t, "tta": tta,
             "dice_mean": round(float(np.mean(values)), 4)}
            for (region, t), values in thr_grid.items()
        ])

    plot_dice_boxplot({r: col[f"dice_{r}"] for r in EVAL_REGIONS}, out_dir / "dice_boxplot.png")
    pred = {r: col[f"vol_pred_{r}"] for r in VOLUME_REGIONS}
    gt = {r: col[f"vol_gt_{r}"] for r in VOLUME_REGIONS}
    plot_bland_altman(pred, gt, out_dir / "bland_altman.png")
    plot_volume_scatter(pred, gt, out_dir / "volume_scatter.png")

    mean_dice = [float(np.mean([r[f"dice_{x}"] for x in EVAL_REGIONS])) for r in main_rows]
    overlay_cases = []
    for label, i in pick_overlay_cases(mean_dice):
        case = cases[i]
        overlay_cases.append({
            "title": f"{label}: {case['id']} (mean Dice {mean_dice[i]:.3f})",
            "image": nib.load(case["flair"]).get_fdata(),
            "gt": np.rint(nib.load(case["seg"]).get_fdata()).astype(np.uint8),
            "pred": np.asarray(nib.load(pred_dir / f"{case['id']}_seg.nii.gz").dataobj),
        })
    plot_case_overlays(overlay_cases, out_dir / "overlays.png", et)
    return summary


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Evaluate a checkpoint on a data split.")
    p.add_argument("--ckpt", required=True, help="best.pt path or Hugging Face model repo id")
    p.add_argument("--filename", default="best.pt", help="File inside the HF repo")
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--splits", type=Path, default=Path("data/splits.json"))
    p.add_argument("--out-root", type=Path, default=Path("results"))
    p.add_argument("--run-name", help="Default: run_name from the checkpoint config")
    p.add_argument("--tta", action="store_true", help="Main results with 8-flip TTA")
    p.add_argument("--ablation", action="store_true", help="TTA off/on x post-processing off/on")
    p.add_argument("--pp-min-component", type=int, default=100,
                   help="Ablation 'postprocess' variant if the config disables it (voxels)")
    p.add_argument("--pp-et-min", type=int, default=500,
                   help="Ablation 'postprocess' variant if the config disables it (voxels)")
    p.add_argument("--region-thresholds", type=float, nargs=3, metavar=("TC", "WT", "ET"),
                   help="Override postprocess.region_thresholds for the main setting")
    p.add_argument("--min-component-voxels", type=int,
                   help="Override postprocess.min_component_voxels for the main setting")
    p.add_argument("--et-min-voxels", type=int,
                   help="Override postprocess.et_min_voxels for the main setting")
    p.add_argument("--pp-grid", action="store_true",
                   help="Also score a grid of post-processing settings (Dice only)")
    p.add_argument("--max-cases", type=int, help="Only the first N cases (smoke tests)")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    predictor = Predictor.from_checkpoint(args.ckpt, device=args.device, filename=args.filename)
    cfg = predictor.config
    if args.region_thresholds is not None:
        cfg.postprocess.region_thresholds = tuple(args.region_thresholds)
    if args.min_component_voxels is not None:
        cfg.postprocess.min_component_voxels = args.min_component_voxels
    if args.et_min_voxels is not None:
        cfg.postprocess.et_min_voxels = args.et_min_voxels
    run_name = args.run_name or cfg.run_name
    if args.split != "test":
        run_name = f"{run_name}_{args.split}"
    cases = select_split(
        discover_cases(args.data_root, require_seg=True), load_splits(args.splits), args.split
    )[: args.max_cases]

    pp_variant = None
    if args.ablation and cfg.postprocess == PostprocessConfig():
        pp_variant = PostprocessConfig(
            min_component_voxels=args.pp_min_component,
            et_min_voxels=args.pp_et_min,
        )
    provenance = {
        "run_name": run_name,
        "checkpoint": str(args.ckpt),
        "model": cfg.model.name,
        "split": args.split,
        "splits_file": str(args.splits),
        "btseg_version": btseg.__version__,
        "git_commit": _git_commit(),
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print(f"Evaluating {cfg.model.name} on {len(cases)} {args.split} cases -> "
          f"{args.out_root / run_name}")
    summary = evaluate(predictor, cases, args.out_root / run_name, args.tta, args.ablation,
                       pp_variant, provenance, args.pp_grid)
    print("Dice " + ", ".join(
        f"{r} {summary['dice'][r]['mean']:.4f}" for r in EVAL_REGIONS
    ) + f" | mean {summary['dice']['mean_of_regions']:.4f}")
    return summary


if __name__ == "__main__":
    main()
