"""SegResNet vs UNet3D on the test split -> results/model_comparison.{json,md}.

Reads only files that evaluate.py / train.py / model_sizes.py already wrote to results/:

- the matched setting for both models is the ``tta+threshold_only`` row of each
  ``results/<run>/ablation.csv`` (TTA on, every region thresholded at 0.5, no other
  post-processing), so the comparison does not depend on the val-tuned WT threshold;
- paired Wilcoxon signed-rank tests use the per-case values in ``per_case.csv``. Those come
  from each run's main evaluation setting; ``settings_match`` says per region whether the two
  runs used the same threshold there (SegResNet's main run uses WT 0.3, so WT is not matched).

    python scripts/compare_models.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

REGIONS = ("WT", "TC", "ET")
MATCHED_VARIANT = "tta+threshold_only"


def read_variant(run_dir: Path, variant: str) -> dict:
    with open(run_dir / "ablation.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["variant"] == variant:
                return row
    raise KeyError(f"{variant} not in {run_dir / 'ablation.csv'}")


def best_val(log_path: Path) -> tuple[float, int, int]:
    log = pd.read_csv(log_path)
    val = log.dropna(subset=["val_dice_mean"])
    best = val.loc[val["val_dice_mean"].idxmax()]
    return float(best["val_dice_mean"]), int(best["epoch"]), int(log["epoch"].max())


def region_thresholds(summary: dict) -> list[float]:
    pp = summary["postprocess"]
    return list(pp["region_thresholds"] or [pp["threshold"]] * 3)  # [TC, WT, ET]


def compare(results: Path, runs: dict[str, str]) -> dict:
    sizes = json.loads((results / "model_sizes.json").read_text())["models"]
    models, per_case, thresholds = {}, {}, {}
    for model, run in runs.items():
        run_dir = results / run
        summary = json.loads((run_dir / "summary.json").read_text())
        row = read_variant(run_dir, MATCHED_VARIANT)
        val_dice, val_epoch, epochs = best_val(results / f"{run}_train_log.csv")
        models[model] = {
            "run": run,
            "parameters": sizes[model]["trainable_parameters"],
            "epochs": epochs,
            "best_val_dice_mean": round(val_dice, 4),
            "best_val_epoch": val_epoch,
            "test_n_cases": summary["n_cases"],
            "test_dice": {r: float(row[f"dice_{r}_mean"]) for r in REGIONS},
            "test_dice_mean_of_regions": float(row["dice_mean_of_regions"]),
            "test_hd95_mm": {r: float(row[f"hd95_{r}_mean"]) for r in REGIONS},
            "wt_volume": {
                k: summary["volume_agreement"]["WT"][k]
                for k in ("mae_cm3", "mean_abs_percent_error", "pearson_r")
            },
            "inference_seconds_per_case": summary["mean_seconds_per_case"],
        }
        per_case[model] = pd.read_csv(run_dir / "per_case.csv").set_index("case_id")
        thresholds[model] = dict(zip(("TC", "WT", "ET"), region_thresholds(summary), strict=True))

    a, b = list(runs)
    common = per_case[a].index.intersection(per_case[b].index)
    tests = {}
    for r in REGIONS:
        for metric in ("dice", "hd95"):
            col = f"{metric}_{r}"
            x, y = per_case[a].loc[common, col], per_case[b].loc[common, col]
            diff = x - y
            a_better = diff > 0 if metric == "dice" else diff < 0  # lower HD95 is better
            b_better = diff < 0 if metric == "dice" else diff > 0
            tests[col] = {
                "mean_difference": round(float(diff.mean()), 4),
                f"cases_{a}_better": int(a_better.sum()),
                f"cases_{b}_better": int(b_better.sum()),
                "ties": int((diff == 0).sum()),
                "wilcoxon_p": float(wilcoxon(x, y).pvalue),
                "settings_match": thresholds[a][r] == thresholds[b][r],
            }
    return {
        "split": "test",
        "matched_setting": f"{MATCHED_VARIANT} (TTA on, threshold 0.5 for every region)",
        "models": models,
        "paired_tests": {"n_cases": len(common), "difference": f"{a} - {b}", "tests": tests},
        "notes": [
            "Inference time is mean_seconds_per_case from each evaluation's summary.json "
            "(separate Kaggle T4 sessions, 8-flip TTA, includes loading the case).",
            "Paired tests use per_case.csv from each run's main evaluation; "
            "settings_match=false means the two runs used different thresholds for that region.",
            "WT volume rows come from each run's main evaluation (summary.json): "
            + ", ".join(f"{n} WT threshold {thresholds[n]['WT']}" for n in runs)
            + ".",
        ],
    }


def to_markdown(c: dict) -> str:
    names = list(c["models"])
    m = c["models"]
    lines = [
        "# SegResNet vs UNet3D (BraTS 2021, test split)",
        "",
        f"Matched setting: {c['matched_setting']}. Generated by `scripts/compare_models.py`.",
        "",
        "| | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]

    def row(label, fn):
        lines.append(f"| {label} | " + " | ".join(fn(m[n]) for n in names) + " |")

    row("Parameters", lambda d: f"{d['parameters'] / 1e6:.2f} M")
    row("Epochs", lambda d: str(d["epochs"]))
    row(
        "Best val mean Dice (epoch)",
        lambda d: f"{d['best_val_dice_mean']:.4f} ({d['best_val_epoch']})",
    )
    for r in REGIONS:
        row(f"Test Dice {r}", lambda d, r=r: f"{d['test_dice'][r]:.4f}")
    row("**Test Dice mean**", lambda d: f"**{d['test_dice_mean_of_regions']:.4f}**")
    for r in REGIONS:
        row(f"Test HD95 {r} (mm)", lambda d, r=r: f"{d['test_hd95_mm'][r]:.2f}")
    row("WT volume MAE (cm³)", lambda d: f"{d['wt_volume']['mae_cm3']:.2f}")
    row("WT volume MAPE (%)", lambda d: f"{d['wt_volume']['mean_abs_percent_error']:.1f}")
    row("WT volume Pearson r", lambda d: f"{d['wt_volume']['pearson_r']:.3f}")
    row("Inference s/case (T4, TTA)", lambda d: f"{d['inference_seconds_per_case']:.1f}")

    t = c["paired_tests"]
    a, b = names
    lines += [
        "",
        f"## Paired per-case tests ({t['n_cases']} cases, difference = {t['difference']})",
        "",
        f"| Metric | Mean difference | {a} better | {b} better | Ties | Wilcoxon p "
        "| Same thresholds |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, v in t["tests"].items():
        lines.append(
            f"| {k} | {v['mean_difference']:+.4f} | {v[f'cases_{a}_better']} "
            f"| {v[f'cases_{b}_better']} | {v['ties']} | {v['wilcoxon_p']:.2g} "
            f"| {'yes' if v['settings_match'] else 'no'} |"
        )
    lines += ["", *[f"- {n}" for n in c["notes"]], ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", type=Path, default=Path("results"))
    args = p.parse_args(argv)
    c = compare(args.results, {"segresnet": "segresnet_dice", "unet3d": "unet3d_dice"})
    (args.results / "model_comparison.json").write_text(json.dumps(c, indent=2) + "\n")
    (args.results / "model_comparison.md").write_text(to_markdown(c), encoding="utf-8")
    print(to_markdown(c))
    return c


if __name__ == "__main__":
    main()
