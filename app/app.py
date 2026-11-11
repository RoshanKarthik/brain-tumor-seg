"""Gradio app: brain tumor segmentation and volumetry from 4 MRI sequences (ZeroGPU-ready).

Weights are loaded once at startup, in this order:
1. ``BTSEG_CHECKPOINT``: path to a local ``best.pt``;
2. ``BTSEG_MODEL_REPO`` or the repo id in ``model_repo.txt`` next to this file
   (written by scripts/build_space.py): ``best.pt`` from that Hugging Face model repo;
3. otherwise an untrained demo model, clearly flagged in the UI (local testing only).

Only the model forward pass runs inside ``@spaces.GPU``; on a machine without ``spaces``
or CUDA everything runs on the CPU, so the same file works locally.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
import time
from pathlib import Path

import gradio as gr
import nibabel as nib
import numpy as np
import torch
from matplotlib import colormaps

from btseg.config import Config
from btseg.data.discover import SEQUENCES
from btseg.data.labels import regions_to_labels
from btseg.infer import Predictor, save_label_map, validate_inputs
from btseg.mesh import segmentation_glb
from btseg.models.factory import build_model
from btseg.postprocess import postprocess
from btseg.viz import AXES, largest_tumor_slice, normalize_for_display, overlay, take_slice
from btseg.volumetrics import VOLUME_REGIONS, region_volumes

try:  # Hugging Face ZeroGPU; a no-op decorator elsewhere
    import spaces

    gpu = spaces.GPU
except ImportError:  # pragma: no cover - spaces is in the app requirements

    def gpu(*args, **kwargs):
        return args[0] if args and callable(args[0]) else (lambda f: f)


APP_DIR = Path(__file__).resolve().parent
DISCLAIMER = (
    "**Research and education only. This is not a medical device and must not be used for "
    "diagnosis, treatment or any clinical decision.** Do not upload identifiable patient data."
)
SEQUENCE_LABELS = {"flair": "FLAIR", "t1": "T1", "t1ce": "T1ce (post-contrast)", "t2": "T2"}


def _model_source() -> tuple[str | None, str]:
    if os.environ.get("BTSEG_CHECKPOINT"):
        return os.environ["BTSEG_CHECKPOINT"], "local checkpoint"
    repo = os.environ.get("BTSEG_MODEL_REPO")
    if not repo and (APP_DIR / "model_repo.txt").is_file():
        repo = (APP_DIR / "model_repo.txt").read_text(encoding="utf-8").strip()
    return (repo, f"Hugging Face model [{repo}](https://huggingface.co/{repo})") if repo else (
        None, "none")


def load_predictor() -> tuple[Predictor, str, bool]:
    """(predictor, description, is_demo)."""
    source, description = _model_source()
    if source:
        return Predictor.from_checkpoint(source, device="cpu"), description, False
    torch.manual_seed(0)
    cfg = Config(run_name="untrained-demo")
    return Predictor(build_model(cfg.model.name, **cfg.model.kwargs), cfg, "cpu"), "none", True


PREDICTOR, MODEL_DESCRIPTION, IS_DEMO = load_predictor()
if torch.cuda.is_available():  # on ZeroGPU this is CUDA emulation outside @spaces.GPU
    PREDICTOR.to("cuda")


def _gpu_seconds(flair, t1, t1ce, t2, tta):
    return 90 if tta else 30


@gpu(duration=_gpu_seconds)
def _predict_maps(flair, t1, t1ce, t2, tta):
    return PREDICTOR.predict_maps([flair, t1, t1ce, t2], tta=bool(tta))


def _path(file) -> str | None:
    if file is None:
        return None
    return file if isinstance(file, str) else getattr(file, "name", None)


def run(flair, t1, t1ce, t2, tta):
    """Validate, segment, and prepare everything the viewer and downloads need."""
    paths = [_path(f) for f in (flair, t1, t1ce, t2)]
    missing = [SEQUENCE_LABELS[s] for s, p in zip(SEQUENCES, paths, strict=True) if not p]
    if missing:
        raise gr.Error(f"Please upload all 4 sequences. Missing: {', '.join(missing)}.")
    try:
        validate_inputs(paths)
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc

    start = time.perf_counter()
    maps = _predict_maps(*paths, tta)
    probabilities = maps.tta_mean if tta else maps.plain
    regions = postprocess(probabilities, PREDICTOR.config.postprocess)
    et_label = PREDICTOR.config.data.et_label
    labels = np.asarray(regions_to_labels(regions, et_label), np.uint8)
    seconds = time.perf_counter() - start
    volumes = region_volumes(labels, maps.affine, et_label)
    device = "GPU" if torch.cuda.is_available() else "CPU"

    raw = {s: nib.load(p).get_fdata() for s, p in zip(SEQUENCES, paths, strict=True)}
    state = {
        "images": {s: normalize_for_display(v).astype(np.float16) for s, v in raw.items()},
        "labels": labels,
        "uncertainty": None if maps.tta_variance is None
        else np.clip(maps.tta_variance.max(axis=0) / 0.25, 0, 1).astype(np.float16),
        "et_label": et_label,
    }

    out_dir = Path(tempfile.mkdtemp(prefix="btseg_"))
    seg_path = out_dir / "segmentation.nii.gz"
    save_label_map(labels, paths[0], seg_path)
    report = {
        "disclaimer": "Research and education only; not for clinical use.",
        "model": PREDICTOR.config.model.name,
        "weights": MODEL_DESCRIPTION if not IS_DEMO else "UNTRAINED DEMO MODEL",
        "tta": bool(tta),
        "device": device,
        "seconds": round(seconds, 2),
        "inputs": {s: Path(p).name for s, p in zip(SEQUENCES, paths, strict=True)},
        "labels": {"1": "NCR (necrotic core)", "2": "ED (edema)", str(et_label): "ET (enhancing)"},
        "volumes": volumes,
    }
    # 3D view: brain outline (nonzero FLAIR voxels) + tumor surfaces, in true mm
    mesh_path = segmentation_glb(labels, maps.affine, out_dir / "tumor_3d.glb", et_label,
                                 brain_mask=raw["flair"] != 0)
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    csv_path = out_dir / "report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["region", "voxels", "volume_cm3"])
        for r in VOLUME_REGIONS:
            writer.writerow([r, volumes[r]["voxels"], round(volumes[r]["cm3"], 3)])

    table = [[r, volumes[r]["voxels"], round(volumes[r]["cm3"], 2)] for r in VOLUME_REGIONS]
    info = (
        f"Done in **{seconds:.1f} s** on **{device}** "
        f"({'with' if tta else 'without'} test-time augmentation)."
    )
    axis = AXES["axial"]
    index = largest_tumor_slice(labels, axis)
    image = render(state, "axial", index, "flair", 0.5, False)
    slider = gr.update(minimum=0, maximum=labels.shape[axis] - 1, value=index, step=1)
    show_unc = gr.update(interactive=state["uncertainty"] is not None,
                         value=False)
    downloads = [str(p) for p in (seg_path, json_path, csv_path, mesh_path) if p]
    model3d = str(mesh_path) if mesh_path else None
    return state, image, slider, show_unc, table, info, downloads, model3d


def render(state, plane, index, sequence, opacity, show_uncertainty):
    if not state:
        return None
    axis = AXES[plane]
    labels = state["labels"]
    index = int(np.clip(index, 0, labels.shape[axis] - 1))
    gray = take_slice(state["images"][sequence].astype(np.float32), axis, index)
    if show_uncertainty and state["uncertainty"] is not None:
        heat = colormaps["inferno"](take_slice(state["uncertainty"].astype(np.float32),
                                               axis, index))[..., :3]
        rgb = (1 - opacity) * np.repeat(gray[..., None], 3, -1) + opacity * heat
    else:
        rgb = overlay(gray, take_slice(labels, axis, index), state["et_label"], opacity)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def change_plane(state, plane, sequence, opacity, show_uncertainty):
    if not state:
        return None, gr.update()
    axis = AXES[plane]
    index = largest_tumor_slice(state["labels"], axis)
    image = render(state, plane, index, sequence, opacity, show_uncertainty)
    return image, gr.update(maximum=state["labels"].shape[axis] - 1, value=index)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Brain tumor segmentation (BraTS MRI)") as demo:
        gr.Markdown("# Brain tumor segmentation and volumetry from MRI")
        gr.Markdown(f"> {DISCLAIMER}")
        if IS_DEMO:
            gr.Markdown(
                "> **UNTRAINED DEMO MODEL:** no trained weights are configured, so the "
                "segmentation is meaningless. Set `BTSEG_CHECKPOINT` or `BTSEG_MODEL_REPO`."
            )
        gr.Markdown(
            "Upload the four co-registered, skull-stripped MRI sequences of one patient as "
            "NIfTI (`.nii` / `.nii.gz`), as in the BraTS challenge. The model segments necrotic "
            "core (NCR, red), edema (ED, green) and enhancing tumor (ET, yellow) and reports "
            "their volumes. MRI only; CT is not supported. Example scans are not bundled "
            "because of the BraTS data terms; BraTS data is available via Synapse after "
            "registration."
        )
        with gr.Row():
            files = [
                gr.File(label=SEQUENCE_LABELS[s], file_types=[".nii", ".gz"], type="filepath")
                for s in SEQUENCES
            ]
        with gr.Row():
            tta = gr.Checkbox(label="Test-time augmentation (8 flips, slower, gives uncertainty)")
            button = gr.Button("Run segmentation", variant="primary")
        info = gr.Markdown()
        state = gr.State()
        with gr.Row():
            with gr.Column(scale=3):
                image = gr.Image(label="Viewer", type="numpy", interactive=False, height=460)
                with gr.Row():
                    plane = gr.Radio(list(AXES), value="axial", label="Plane")
                    sequence = gr.Radio(list(SEQUENCES), value="flair", label="Sequence")
                index = gr.Slider(0, 154, value=77, step=1, label="Slice")
                with gr.Row():
                    opacity = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Overlay opacity")
                    show_unc = gr.Checkbox(label="Show uncertainty (needs TTA)", interactive=False)
            with gr.Column(scale=2):
                table = gr.Dataframe(headers=["Region", "Voxels", "Volume (cm³)"],
                                     label="Volumes (TC = NCR + ET, WT = NCR + ED + ET)",
                                     interactive=False)
                downloads = gr.File(label="Downloads: segmentation.nii.gz, report.json, "
                                    "report.csv, tumor_3d.glb",
                                    file_count="multiple", interactive=False)
                gr.Markdown(f"Model weights: {MODEL_DESCRIPTION}. Model: "
                            f"`{PREDICTOR.config.model.name}`.")
        gr.Markdown(
            "### 3D view\n"
            "Drag to rotate, scroll to zoom. Necrotic core red, enhancing tumor yellow, "
            "edema / whole tumor translucent green, brain outline faint gray. Viewed from "
            "the front: the patient's right is on your left. True to scale (mm)."
        )
        model3d = gr.Model3D(label="3D tumor view", clear_color=(0.08, 0.08, 0.1, 1.0),
                             height=520, interactive=False)

        button.click(run, [*files, tta],
                     [state, image, index, show_unc, table, info, downloads, model3d])
        view_inputs = [state, plane, index, sequence, opacity, show_unc]
        for control in (index, sequence, opacity, show_unc):
            control.change(render, view_inputs, image)
        plane.change(change_plane, [state, plane, sequence, opacity, show_unc], [image, index])
    return demo


if __name__ == "__main__":
    build_ui().launch()
