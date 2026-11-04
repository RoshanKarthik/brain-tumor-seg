"""Resumable training loop.

    python -m btseg.train --config configs/segresnet.yaml --data-root /data/brats \\
        --out runs/segresnet_dice [--resume] [--epochs N] [--max-train-cases N]

Writes to ``--out``:
- ``last.pt``: model, optimizer, scheduler, AMP scaler, epoch, best metric (every epoch)
- ``best.pt``: model weights + config of the epoch with the best mean val Dice
- ``train_log.csv``: one row per epoch
- ``config.json``: the resolved config

Kaggle sessions end after 12 h, so ``--time-limit-hours`` stops cleanly between epochs
and ``--resume`` continues from ``last.pt``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from monai.data import DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.utils import set_determinism

import btseg
from btseg.config import Config, load_config
from btseg.data.dataset import build_dataset, load_splits, select_split
from btseg.data.discover import discover_cases
from btseg.data.labels import REGIONS
from btseg.data.transforms import get_train_transforms, get_val_transforms
from btseg.losses import build_loss
from btseg.models.factory import build_model, check_roi_size

LOG_FIELDS = [
    "epoch",
    "lr",
    "train_loss",
    *(f"val_dice_{r.lower()}" for r in REGIONS),
    "val_dice_mean",
    "best_val_dice",
    "train_seconds",
    "data_wait_seconds",  # part of train_seconds spent waiting for the DataLoader (CPU-bound)
    "val_seconds",
    "epoch_seconds",
]


def seed_everything(seed: int) -> None:
    # seeds python, numpy, torch and MONAI's random transforms; keep cuDNN autotuning
    set_determinism(seed=seed, use_deterministic_algorithms=False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def lr_factor(epoch: int, epochs: int, warmup_epochs: int) -> float:
    """Linear warmup over ``warmup_epochs``, then cosine decay to 0 at ``epochs``."""
    if epoch < warmup_epochs:
        return (epoch + 1) / (warmup_epochs + 1)
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _atomic_save(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _append_log(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)


def train_one_epoch(
    model, loader, loss_fn, optimizer, scaler, device, amp: bool
) -> tuple[float, float]:
    """Return (mean loss, seconds spent waiting for batches)."""
    model.train()
    total, n, data_wait = 0.0, 0, 0.0
    t_wait = time.perf_counter()
    for batch in loader:
        data_wait += time.perf_counter() - t_wait
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            loss = loss_fn(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * images.shape[0]  # .item() syncs, so the wait below is honest
        n += images.shape[0]
        t_wait = time.perf_counter()
    return total / max(n, 1), data_wait


@torch.no_grad()
def validate(model, loader, cfg: Config, device, amp: bool) -> list[float]:
    """Mean Dice per region [TC, WT, ET] over the loader (sigmoid > 0.5).

    Uses the BraTS convention for empty ground truth: Dice 1 if the prediction is also
    empty, else 0 (``ignore_empty=False``).
    """
    model.eval()
    metric = DiceMetric(include_background=True, reduction="mean_batch", ignore_empty=False)
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = sliding_window_inference(
                images,
                roi_size=cfg.infer.roi_size,
                sw_batch_size=cfg.infer.sw_batch_size,
                predictor=model,
                overlap=cfg.infer.overlap,
                mode=cfg.infer.mode,
            )
        metric(y_pred=(torch.sigmoid(logits.float()) > 0.5).float(), y=labels)
    return [float(v) for v in metric.aggregate().cpu()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a BraTS segmentation model.")
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--splits", type=Path, help="Defaults to data.splits from the config")
    p.add_argument("--resume", action="store_true", help="Continue from <out>/last.pt")
    p.add_argument("--epochs", type=int, help="Override train.epochs")
    p.add_argument("--max-train-cases", type=int, help="Use only the first N train cases")
    p.add_argument("--max-val-cases", type=int, help="Use only the first N val cases")
    p.add_argument(
        "--time-limit-hours",
        type=float,
        help="Stop between epochs before this wall-clock budget runs out",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    check_roi_size(cfg.model.name, cfg.data.roi_size)
    check_roi_size(cfg.model.name, cfg.infer.roi_size)

    start_time = time.time()
    seed_everything(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = cfg.train.amp and device.type == "cuda"

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    last_path, best_path, log_path = out / "last.pt", out / "best.pt", out / "train_log.csv"
    if args.resume and not last_path.exists():
        raise FileNotFoundError(f"--resume given but {last_path} does not exist")
    if not args.resume and last_path.exists():
        raise FileExistsError(f"{last_path} exists; pass --resume or use a new --out directory")
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n", encoding="utf-8")

    # data
    splits_path = args.splits or Path(cfg.data.splits)
    if not splits_path.exists():
        raise FileNotFoundError(
            f"Split file {splits_path} not found; create it with scripts/make_splits.py"
        )
    splits = load_splits(splits_path)
    cases = discover_cases(args.data_root, require_seg=True)
    train_cases = select_split(cases, splits, "train")[: args.max_train_cases]
    val_cases = select_split(cases, splits, "val")[: args.max_val_cases]
    train_tf = get_train_transforms(
        cfg.data.roi_size, cfg.data.et_label, cfg.data.pixdim, cfg.data.crops_per_case
    )
    train_loader = DataLoader(
        build_dataset(train_cases, train_tf),
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_cases) > cfg.data.batch_size,
        persistent_workers=cfg.data.num_workers > 0,
    )
    val_loader = DataLoader(
        build_dataset(val_cases, get_val_transforms(cfg.data.et_label, cfg.data.pixdim)),
        batch_size=1,
        num_workers=cfg.data.num_workers,
        pin_memory=device.type == "cuda",
    )

    # model / optimization
    model = build_model(cfg.model.name, **cfg.model.kwargs).to(device)
    loss_fn = build_loss(cfg.loss.name, **cfg.loss.kwargs)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: lr_factor(e, cfg.train.epochs, cfg.train.warmup_epochs)
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    start_epoch, best_metric, best_epoch = 0, -1.0, -1
    if args.resume:
        ckpt = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"]
        best_metric, best_epoch = ckpt["best_metric"], ckpt["best_epoch"]
        print(f"Resumed from {last_path}: {start_epoch} epoch(s) done, best {best_metric:.4f}")

    print(
        f"{cfg.run_name}: {cfg.model.name} on {device} (AMP {amp}), "
        f"{len(train_cases)} train / {len(val_cases)} val cases, "
        f"epochs {start_epoch + 1}..{cfg.train.epochs}"
    )

    epochs_done = start_epoch
    last_epoch_seconds = 0.0
    for epoch in range(start_epoch, cfg.train.epochs):
        if args.time_limit_hours is not None:
            elapsed = time.time() - start_time
            if elapsed + 1.2 * last_epoch_seconds > args.time_limit_hours * 3600:
                print(
                    f"Stopping before epoch {epoch + 1}: time limit would be exceeded. "
                    "Rerun with --resume to continue."
                )
                break
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, data_wait = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scaler, device, amp
        )
        train_seconds = time.time() - t0
        scheduler.step()

        done = epochs_done = epoch + 1
        dice = None
        t_val = time.time()
        if done % cfg.train.val_interval == 0 or done == cfg.train.epochs:
            dice = validate(model, val_loader, cfg, device, amp)
            mean_dice = float(np.mean(dice))
            if mean_dice > best_metric:
                best_metric, best_epoch = mean_dice, done
                _atomic_save(
                    {
                        "model": model.state_dict(),
                        "config": cfg.to_dict(),
                        "epoch": done,
                        "val_dice": dict(zip(REGIONS, dice, strict=True)),
                        "val_dice_mean": mean_dice,
                        "btseg_version": btseg.__version__,
                    },
                    best_path,
                )

        val_seconds = time.time() - t_val
        last_epoch_seconds = time.time() - t0
        _atomic_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": done,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "config": cfg.to_dict(),
                "btseg_version": btseg.__version__,
            },
            last_path,
        )
        row = {
            "epoch": done,
            "lr": f"{lr:.3e}",
            "train_loss": f"{train_loss:.5f}",
            "best_val_dice": f"{best_metric:.5f}" if best_epoch > 0 else "",
            "train_seconds": f"{train_seconds:.1f}",
            "data_wait_seconds": f"{data_wait:.1f}",
            "val_seconds": f"{val_seconds:.1f}" if dice is not None else "",
            "epoch_seconds": f"{last_epoch_seconds:.1f}",
        }
        line = f"epoch {done}/{cfg.train.epochs} | loss {train_loss:.4f} | lr {lr:.2e}"
        if dice is not None:
            for region, value in zip(REGIONS, dice, strict=True):
                row[f"val_dice_{region.lower()}"] = f"{value:.5f}"
            row["val_dice_mean"] = f"{np.mean(dice):.5f}"
            line += " | val " + " ".join(
                f"{r} {v:.3f}" for r, v in zip(REGIONS, dice, strict=True)
            )
            line += f" mean {np.mean(dice):.3f} (best {best_metric:.3f} @ {best_epoch})"
        _append_log(log_path, row)
        print(
            f"{line} | {last_epoch_seconds:.0f}s (train {train_seconds:.0f}s, "
            f"waiting for data {data_wait:.0f}s)",
            flush=True,
        )

    return {"epochs_done": epochs_done, "best_metric": best_metric, "best_epoch": best_epoch}


if __name__ == "__main__":
    main()
