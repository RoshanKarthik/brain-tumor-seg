"""Overlays and evaluation plots (matplotlib object API, no display backend needed).

Overlay colors: NCR red, ED green, ET yellow (shared with the app).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

from btseg.data.labels import ED_LABEL, NCR_LABEL

COLORS = {"NCR": (1.0, 0.0, 0.0), "ED": (0.0, 0.8, 0.0), "ET": (1.0, 0.9, 0.0)}
AXES = {"sagittal": 0, "coronal": 1, "axial": 2}


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    """Scale to [0, 1] using the 1st-99th percentile of nonzero voxels."""
    image = np.asarray(image, dtype=np.float32)
    nonzero = image[image != 0]
    if nonzero.size == 0:
        return np.zeros_like(image)
    lo, hi = np.percentile(nonzero, [1, 99])
    return np.clip((image - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def overlay(image2d, labels2d=None, et_label: int = 4, alpha: float = 0.5) -> np.ndarray:
    """Gray image (already in [0, 1]) with colored labels -> (H, W, 3) float RGB."""
    gray = np.asarray(image2d, dtype=np.float32)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    if labels2d is None:
        return rgb
    labels2d = np.asarray(labels2d)
    for name, value in (("NCR", NCR_LABEL), ("ED", ED_LABEL), ("ET", et_label)):
        mask = labels2d == value
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * np.asarray(COLORS[name])
    return rgb


def largest_tumor_slice(labels: np.ndarray, axis: int = 2) -> int:
    """Index of the slice with the most tumor voxels along ``axis`` (middle if no tumor)."""
    counts = (np.asarray(labels) > 0).sum(axis=tuple(a for a in range(3) if a != axis))
    return int(np.argmax(counts)) if counts.any() else labels.shape[axis] // 2


def take_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    """2D slice rotated for display (rows = second remaining axis, top = superior/anterior)."""
    return np.rot90(np.take(volume, index, axis=axis))


def plot_case_overlays(cases: Sequence[dict], path: str | Path, et_label: int = 4) -> None:
    """One row per case: FLAIR, ground truth overlay, prediction overlay (axial slice).

    Each case dict has ``title``, ``image`` (3D FLAIR), ``gt`` and ``pred`` (3D labels).
    """
    fig = Figure(figsize=(9, 3 * len(cases)), layout="constrained")
    axes = fig.subplots(len(cases), 3, squeeze=False)
    for row, case in zip(axes, cases, strict=True):
        k = largest_tumor_slice(case["gt"], axis=2)
        img = take_slice(normalize_for_display(case["image"]), 2, k)
        panels = [
            (overlay(img), f"{case['title']}\nFLAIR, axial {k}"),
            (overlay(img, take_slice(case["gt"], 2, k), et_label), "ground truth"),
            (overlay(img, take_slice(case["pred"], 2, k), et_label), "prediction"),
        ]
        for ax, (rgb, title) in zip(row, panels, strict=True):
            ax.imshow(rgb)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    legend = ", ".join(f"{n} {c}" for n, c in zip(COLORS, ("red", "green", "yellow"), strict=True))
    fig.suptitle(f"Colors: {legend}", fontsize=9)
    fig.savefig(path, dpi=120)


def plot_dice_boxplot(dice: dict[str, Sequence[float]], path: str | Path) -> None:
    fig = Figure(figsize=(5, 4), layout="constrained")
    ax = fig.subplots()
    names = list(dice)
    ax.boxplot([dice[n] for n in names], tick_labels=names, showmeans=True)
    ax.set_ylabel("Dice")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Dice per region (n = {len(dice[names[0]])})")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(path, dpi=120)


def plot_bland_altman(
    pred: dict[str, np.ndarray], gt: dict[str, np.ndarray], path: str | Path
) -> None:
    """Difference (pred - GT) vs. mean volume per region, with bias and 95% limits."""
    regions = list(pred)
    fig = Figure(figsize=(4 * len(regions), 3.6), layout="constrained")
    axes = fig.subplots(1, len(regions), squeeze=False)[0]
    for ax, r in zip(axes, regions, strict=True):
        p, g = np.asarray(pred[r]), np.asarray(gt[r])
        mean, diff = (p + g) / 2, p - g
        bias, sd = diff.mean(), diff.std(ddof=1) if len(diff) > 1 else 0.0
        ax.scatter(mean, diff, s=10, alpha=0.7)
        for y, style in ((bias, "-"), (bias + 1.96 * sd, "--"), (bias - 1.96 * sd, "--")):
            ax.axhline(y, color="k", linestyle=style, linewidth=1)
        ax.set_title(f"{r}: bias {bias:+.2f}, LoA ±{1.96 * sd:.2f} cm³", fontsize=9)
        ax.set_xlabel("mean of predicted and GT volume (cm³)")
        ax.set_ylabel("predicted − GT (cm³)")
        ax.grid(alpha=0.3)
    fig.savefig(path, dpi=120)


def plot_volume_scatter(
    pred: dict[str, np.ndarray], gt: dict[str, np.ndarray], path: str | Path
) -> None:
    regions = list(pred)
    fig = Figure(figsize=(4 * len(regions), 3.8), layout="constrained")
    axes = fig.subplots(1, len(regions), squeeze=False)[0]
    for ax, r in zip(axes, regions, strict=True):
        p, g = np.asarray(pred[r]), np.asarray(gt[r])
        hi = max(float(p.max(initial=0)), float(g.max(initial=0)), 1e-3) * 1.05
        ax.plot([0, hi], [0, hi], "k--", linewidth=1)
        ax.scatter(g, p, s=10, alpha=0.7)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_aspect("equal")
        ax.set_title(r, fontsize=9)
        ax.set_xlabel("ground-truth volume (cm³)")
        ax.set_ylabel("predicted volume (cm³)")
        ax.grid(alpha=0.3)
    fig.savefig(path, dpi=120)
