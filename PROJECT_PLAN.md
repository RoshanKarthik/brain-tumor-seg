# Project Plan: Brain Tumor Segmentation & Volumetric Quantification

> **For Claude Code:** This file is the source of truth for the project. Read it fully before writing code. Work **one phase at a time**, in order. Finish each phase's acceptance checks, commit, then summarize what was done and what the human must do next before moving on. If something in this plan turns out to be wrong or outdated (library APIs, platform limits), stop and tell the human instead of silently working around it.

---

## 1. Goal

An end-to-end deep learning system that:

1. Takes a patient's multi-sequence brain MRI (FLAIR, T1, T1ce, T2 as NIfTI files).
2. Segments the tumor into three sub-regions: **necrotic core (NCR)**, **peritumoral edema (ED)** and **enhancing tumor (ET)**.
3. Computes the volume (cm³ / mL) of each region and of the standard composite regions: **Whole Tumor (WT)**, **Tumor Core (TC)**, **Enhancing Tumor (ET)**.
4. Is deployed as a free, public web app where anyone can upload scans and get the segmentation overlay, volumes, and downloadable results.

Academic context: semester project for *Deep Learning for Computer Vision*. The deliverables are working code, a deployed demo, an experiments/results section with real numbers, and a written report.

### Scope note on CT
Public, labeled CT brain-tumor segmentation datasets are scarce. This system is trained and evaluated on **MRI only**. CT is discussed in the report as a limitation and future work (e.g., transfer learning once labeled CT is available). **Do not attempt CT training or claim CT support in the app.**

---

## 2. Hard constraints

- **Everything must be free.** No paid APIs, paid hosting, or paid GPU. If a step seems to need money, stop and tell the human.
- **No fabricated results.** Every number in the README, report, model card or app must come from files produced by `evaluate.py`. Until those exist, use clearly marked placeholders like `TBD`.
- **Not a medical device.** The app must show a clear disclaimer: research/education only, not for clinical use.
- **Local development has no GPU and no dataset.** All tests run on CPU with small synthetic volumes. Real training happens on Kaggle.

---

## 3. Free platform stack

| Purpose | Platform | Notes |
|---|---|---|
| Code hosting + CI | GitHub (public repo) + GitHub Actions | CI runs pytest on CPU |
| Training GPU | **Kaggle Notebooks** | ~30 GPU-hours/week quota (T4 or P100), 12-hour session limit. Training must be resumable from checkpoints. |
| Dataset | **BraTS 2021 Task 1** (1,251 labeled cases) on Kaggle | Attach to the Kaggle notebook as an input dataset |
| Model weights | Hugging Face Hub (public model repo) | Free |
| App hosting (primary) | **Hugging Face Space, Gradio SDK, ZeroGPU hardware** | As of mid-2026, free accounts can no longer create CPU-Basic Gradio/Docker Spaces; free personal accounts *in good standing* can host up to 2 ZeroGPU Gradio Spaces. Visitors get a small daily GPU quota (a few minutes), which is plenty for a few-second 3D inference. |
| App hosting (fallback) | Streamlit Community Cloud, CPU-only with ONNX Runtime | Only if ZeroGPU is unavailable on the human's account. Memory is limited (~1–2.7 GB), so inference must be lean. |

**Check current platform docs before Phase 7.** These free tiers change often.

---

## 4. Data

**BraTS 2021 Task 1** training set: 1,251 cases, each with:
`BraTS2021_XXXXX_flair.nii.gz`, `_t1.nii.gz`, `_t1ce.nii.gz`, `_t2.nii.gz`, `_seg.nii.gz`
Volumes are 240 × 240 × 155, already co-registered, skull-stripped and resampled to 1 mm isotropic.

**Labels in BraTS 2021:** 0 = background, 1 = NCR, 2 = ED, **4 = ET**. (BraTS 2023+ uses 3 for ET, so make the ET label value a config option.)

**Composite regions (model targets, multi-label, 3 sigmoid channels):**
- TC = label ∈ {1, ET}
- WT = label ∈ {1, 2, ET}
- ET = label == ET

Write this conversion explicitly in our own code with a unit test. Do not rely on library defaults whose label assumptions may differ.

**Converting predictions back to labels** (for display and NIfTI export) uses nested hierarchy: start from 0; WT → 2; TC → 1; ET → ET label. This guarantees ET ⊆ TC ⊆ WT.

**Split:** a fixed seed (42) patient-level split, 80 / 10 / 10 → train / val / test. Save it to `data/splits.json` and commit it. The test set is used **only once**, for final evaluation.

**Kaggle storage:** the output directory `/kaggle/working` is limited (~20 GB). If the dataset arrives as a `.tar`, extract it to a scratch location outside `/kaggle/working`. The loader must accept any root directory and find cases by recursive glob.

---

## 5. Tech stack

- Python 3.10 or 3.11
- PyTorch + **MONAI** (models, transforms, sliding-window inference, metrics)
- nibabel, numpy, scipy, pandas, matplotlib
- Gradio (app), `spaces` package (ZeroGPU decorator)
- onnx + onnxruntime (CPU fallback export)
- pytest, ruff
- huggingface_hub (upload weights, download in app)

Pin versions in `requirements.txt` after confirming that they install together. The app has its own minimal `app/requirements.txt`. Check the ZeroGPU docs for supported torch versions.

---

## 6. Repository layout

```
brain-tumor-seg/
├── PROJECT_PLAN.md              # this file
├── README.md
├── requirements.txt             # training / dev
├── pyproject.toml               # package "btseg", installable with pip install -e .
├── configs/
│   ├── segresnet.yaml           # main model
│   └── unet3d.yaml              # baseline
├── data/
│   └── splits.json
├── src/btseg/
│   ├── config.py                # YAML -> dataclass
│   ├── data/
│   │   ├── discover.py          # find cases under a root dir
│   │   ├── labels.py            # label <-> region conversion
│   │   ├── transforms.py        # train / val / infer pipelines
│   │   └── dataset.py
│   ├── models/factory.py        # build_model(name, **kwargs)
│   ├── losses.py
│   ├── train.py                 # resumable training loop
│   ├── infer.py                 # Predictor class: files -> mask + probs
│   ├── postprocess.py
│   ├── metrics.py               # Dice, HD95 with BraTS edge-case handling
│   ├── volumetrics.py           # voxel volume from affine, per-region cm^3
│   ├── evaluate.py              # test-set eval -> CSV/JSON + figures
│   ├── viz.py                   # overlays, plots
│   └── export.py                # ONNX export + parity check
├── scripts/
│   ├── make_splits.py
│   ├── upload_weights.py        # push checkpoint + model card to HF Hub
│   └── build_space.py           # assemble app/ + btseg into a Space folder and upload
├── notebooks/
│   └── kaggle_train.ipynb       # thin wrapper: clone repo, pip install, run train/eval
├── app/
│   ├── app.py                   # Gradio UI (ZeroGPU-ready)
│   ├── requirements.txt
│   └── README.md                # HF Space YAML header
├── tests/                       # all CPU, synthetic data
├── results/                     # filled by evaluate.py (committed CSV/JSON/PNGs)
└── report/
    └── report.md                # academic write-up
```

---

## 7. Phases

### Phase 0: Scaffold
- Create the repo layout, `pyproject.toml`, `requirements.txt`, ruff config, `.gitignore` (ignore data, checkpoints, `*.nii.gz` except tiny test fixtures).
- Add GitHub Actions CI that installs the CPU-only torch wheel and runs `ruff` + `pytest`.
- **Accept:** `pip install -e .` works, `pytest` runs (even if empty), CI is green.

### Phase 1: Data pipeline
- `discover.py`: given a root, return a list of `{id, flair, t1, t1ce, t2, seg?}`. Fail loudly on missing sequences.
- `labels.py`: `labels_to_regions(seg, et_label=4)` → (3, H, W, D) in order [TC, WT, ET]; `regions_to_labels(regions, et_label=4)` using the hierarchy in §4.
- `transforms.py` (MONAI dictionary transforms):
  - All: load, ensure channel-first, stack the 4 sequences in the fixed order **[FLAIR, T1, T1ce, T2]**, orient to RAS, resample to 1 mm (no-op for BraTS but needed for user uploads), per-channel z-score normalization on nonzero (brain) voxels.
  - Train: crop foreground, random spatial crop **128³** (configurable to 96³), random flips on all 3 axes, small random intensity scale/shift.
  - Val/infer: no cropping; whole volume goes through sliding-window inference.
- `make_splits.py` → `data/splits.json`.
- **Tests:** label round-trip (labels → regions → labels equals original), ET ⊆ TC ⊆ WT, transforms produce expected shapes on a synthetic 4-channel 64³ volume, and discovery works on a fake directory tree.
- **Accept:** tests pass on CPU.

### Phase 2: Models and loss
- `factory.py` supports:
  - `segresnet`: MONAI SegResNet, `in_channels=4, out_channels=3`, `init_filters=16`, `blocks_down=(1,2,2,4)`, `blocks_up=(1,1,1)`, `dropout_prob=0.2`. This is the **main model**: light and fast enough for ZeroGPU and the CPU fallback.
  - `unet3d`: MONAI UNet 3D baseline with comparable parameter count.
  - `swinunetr`: optional, only if the GPU budget allows (see §8).
- `losses.py`: `DiceLoss(sigmoid=True)` and `DiceFocalLoss(sigmoid=True)`, selected by config.
- **Tests:** forward pass on a (1, 4, 64, 64, 64) tensor returns (1, 3, 64, 64, 64) for each model; print parameter counts.
- **Accept:** tests pass; parameter counts recorded in `results/model_sizes.json`.

### Phase 3: Training (runs on Kaggle)
- `train.py --config configs/segresnet.yaml --data-root ... --out ...`
  - AdamW (lr 1e-4, wd 1e-5), cosine schedule with warmup, **AMP**, batch size 2 (configurable), `num_workers` from config.
  - Do not cache the whole dataset in RAM (~140 MB per case as float32). Use a plain `Dataset` with workers.
  - Validate every N epochs on the val split with sliding-window inference (roi 128³, overlap 0.5). Track mean Dice per region.
  - **Checkpointing:** save `last.pt` every epoch (model, optimizer, scheduler, scaler, epoch, best metric) and `best.pt` by mean val Dice. `--resume` continues from `last.pt`. This is required because Kaggle sessions end at 12 h.
  - Log to CSV (`train_log.csv`) and print a compact per-epoch line. No paid trackers.
  - Seed everything.
  - `--max-train-cases` and `--epochs` flags for quick smoke runs.
- `notebooks/kaggle_train.ipynb`: cells to (1) clone the GitHub repo and `pip install -e .`, (2) locate or extract the BraTS data, (3) run a 2-epoch smoke test on 20 cases, (4) run full training with resume, (5) optionally upload `best.pt` to HF Hub using a token from **Kaggle Secrets** (`HF_TOKEN`), never hard-coded.
- **Local test:** a 1-epoch run on 2 synthetic cases at 64³ completes on CPU and writes checkpoints; resume works.
- **Accept:** local smoke test passes. Then hand off to the human with exact Kaggle steps (see §10).

### Phase 4: Inference, post-processing, volumetrics
- `infer.py`: a `Predictor` class
  - `Predictor.from_checkpoint(path_or_hf_repo, device="auto")`
  - `predict(flair, t1, t1ce, t2) -> PredictionResult` with: the label map **in the original image space** (use MONAI inverse transforms so the output aligns with the uploaded files), region probabilities, the affine, and inference time.
  - Sliding-window inference (roi 128³, overlap 0.5; sw_batch_size 2 on GPU, 1 on CPU). Use AMP on GPU.
  - Optional flip **TTA** (8 flips): average the probabilities and use the per-voxel variance as an **uncertainty map**.
- `postprocess.py`: threshold 0.5, enforce the hierarchy, optional removal of small connected components, optional "ET-small → reassign to NCR" rule (if predicted ET < k voxels). Each step is toggled by config so it can be ablated.
- `volumetrics.py`:
  - Voxel volume in mm³ = `abs(det(affine[:3, :3]))`, not an assumed 1 mm³.
  - Return the NCR, ED, ET, TC and WT volumes in cm³ (= mL), plus voxel counts.
- **Tests:** a synthetic sphere of known radius gives the expected volume within 2%; anisotropic spacing (e.g., 0.5 × 0.5 × 2 mm) scales correctly; the inverse transform returns the original shape.
- **Accept:** tests pass; `python -m btseg.infer --ckpt ... --case-dir ...` writes `seg.nii.gz` and `volumes.json`.

### Phase 5: Evaluation (runs on Kaggle after training)
- `evaluate.py --ckpt ... --split test` writes to `results/<run_name>/`:
  - `per_case.csv`: Dice and HD95 for WT/TC/ET, predicted vs. ground-truth volume per region.
  - `summary.json`: mean ± std, median.
  - Volume agreement: MAE (cm³), mean absolute % error, Pearson r, plus **Bland–Altman** and **scatter (pred vs GT)** plots per region.
  - Figures: Dice box plots per region, 6 qualitative overlays (best, median and worst cases).
- **Metric edge cases (BraTS convention):** if both GT and prediction are empty for a region, Dice = 1 and HD95 = 0. If exactly one is empty, Dice = 0 and HD95 = 373.13 mm. Document this.
- **Ablations** (inference-only, cheap, no retraining): TTA on/off, post-processing on/off. Training comparison: SegResNet vs. UNet3D, and Dice vs. DiceFocal loss if the budget allows.
- **Accept:** running on a synthetic mini-split works locally; the real numbers come from Kaggle and are committed to `results/`.

### Phase 6: Export
- `export.py`: export the best SegResNet to ONNX (dynamic spatial dims, or a fixed 128³ patch with sliding window done in numpy/torch on CPU). Verify parity with PyTorch on a random input (max abs diff < 1e-3).
- `scripts/upload_weights.py`: upload `best.pt`, `model.onnx`, the config, and a **model card** (architecture, data, metrics read from `summary.json`, intended use, limitations, data-license note) to an HF Hub model repo.
- **Accept:** parity test passes; upload script works with a dry-run flag.

### Phase 7: Web app (Gradio on HF ZeroGPU)
- `app/app.py`:
  - **Inputs:** 4 file uploads (FLAIR, T1, T1ce, T2; `.nii` / `.nii.gz`), a TTA checkbox, and a "Run segmentation" button. Optionally one-click example cases if a license-compatible sample is available (see §10).
  - **Validation:** all 4 present, same shape, affines approximately equal, 3D. Show friendly errors.
  - **Inference:** call `Predictor` inside a function decorated with `@spaces.GPU(duration=...)`. It must still run on CPU when `spaces`/CUDA is unavailable, so the same app works locally.
  - **Outputs:**
    - A slice viewer with axial / coronal / sagittal choice and a slice slider, showing the selected sequence with a color overlay (NCR red, ED green, ET yellow) and an opacity slider. Start at the slice with the largest tumor area. Store the volumes in `gr.State` so slider moves don't rerun the model.
    - Optional uncertainty heatmap view (when TTA is on).
    - Volume table: NCR, ED, ET, TC, WT in cm³.
    - Downloads: `segmentation.nii.gz` (BraTS labels, original space) and `report.json` / `report.csv`.
    - Inference time and device used.
  - The disclaimer is visible on the page.
  - Weights load once at startup via `hf_hub_download` from the model repo.
- `app/README.md`: HF Space YAML header (`sdk: gradio`, title, emoji, `app_file: app.py`, hardware set to ZeroGPU per current docs, linked model repo).
- `scripts/build_space.py`: copy `app/` plus the `src/btseg` package into `build/space/`, then upload it with `huggingface_hub` (`upload_folder`). The human runs it after `huggingface-cli login`.
- **Tests:** an app smoke test that runs the predict function on CPU with a tiny randomly initialized model and synthetic NIfTI files written to a temp dir.
- **Accept:** `python app/app.py` runs locally on CPU end-to-end with a real checkpoint (if downloaded) or a dummy model.

### Phase 7b: Fallback deployment (only if ZeroGPU is unavailable)
- `app_streamlit/app.py`: same features in a lighter form, using **ONNX Runtime on CPU**, no TTA by default, aggressive memory hygiene (free arrays, float16 storage for display volumes), and deployed from GitHub to Streamlit Community Cloud.
- Keep all logic in `btseg`; each UI is a thin layer.

### Phase 8: Documentation and report
- `README.md`: overview, demo link, architecture diagram (Mermaid), results table (from `summary.json`), how to train on Kaggle, how to run locally, how to deploy, limitations, disclaimer, citations (BraTS papers, MONAI, SegResNet).
- `report/report.md` sections: Abstract, Introduction (clinical motivation for volumetrics), Related Work (U-Net, 3D U-Net, SegResNet, nnU-Net, Swin-UNETR), Dataset, Method (preprocessing, architecture, loss, training, inference, TTA, post-processing, volume computation), Experiments (setup, metrics, baselines, ablations), Results (tables + figures from `results/`), Discussion (failure cases, volume error vs. tumor size, ET difficulty), Limitations (MRI-only, gliomas only, pre-operative scans only, no external validation, CT not covered), Future Work, Conclusion, References.
- **Accept:** every number in the README and report traces back to a file in `results/`.

---

## 8. GPU budget (Kaggle, ~30 h/week)

| Run | Est. time | Priority |
|---|---|---|
| Smoke test (20 cases, 2 epochs) | < 15 min | Required |
| SegResNet, DiceLoss, ~100–150 epochs | ~12–20 h (2 sessions with resume) | Required (main model) |
| Test-set evaluation + TTA ablation | ~1 h | Required |
| UNet3D baseline, same schedule (or 50% of the epochs if short on time) | ~8–15 h | Required for comparison |
| SegResNet + DiceFocal | ~12 h | Optional |
| SwinUNETR | ~20 h+ | Optional, only with spare weeks |

Measure the real time per epoch in the smoke test and adjust the epoch counts. Run one GPU notebook at a time (they share the quota). Use "Save Version → Run All" so training continues in the background.

---

## 9. Suggested timeline

| Week | Work |
|---|---|
| 1 | Phases 0–2 (Claude Code); human sets up accounts (§10) |
| 2 | Phase 3 locally; smoke test + start main training on Kaggle |
| 3 | Finish main training; baseline training; Phase 4 |
| 4 | Phase 5 evaluation; Phase 6 export + upload |
| 5 | Phase 7 app + deploy |
| 6 | Phase 8 report, polish, demo rehearsal |

Compress or stretch this to fit the actual deadline.

---

## 10. Things only the human can do

Claude Code should prompt the human at the right phase. It must never handle passwords or create accounts.

1. **Now:** create free accounts on GitHub, Kaggle, and Hugging Face.
   - Kaggle: verify your phone number (needed for GPU and internet in notebooks).
   - Hugging Face: verify your email early. ZeroGPU hosting on free accounts reportedly requires an account in good standing, and possibly a minimum account age.
2. Create a Hugging Face **write token**. Add it as a Kaggle Secret named `HF_TOKEN`, and use `huggingface-cli login` locally. Never commit it.
3. On Kaggle: create a notebook from `notebooks/kaggle_train.ipynb`, attach the BraTS 2021 Task 1 dataset, set the accelerator to GPU, turn internet on, and run it.
4. Check the BraTS data terms before redistributing any scan (e.g., as example cases in the public app). If redistribution isn't clearly allowed, don't bundle samples; tell users where to download the data instead.
5. Create the HF model repo and Space (or let the scripts create them after login), then run `scripts/build_space.py`.
6. Confirm with the instructor that an MRI-only system is acceptable for the "MRI/CT" title.

---

## 11. Definition of done

- [ ] CI green; tests cover labels, transforms, models, volumetrics, metrics edge cases, inference inverse-mapping, and the app smoke test
- [ ] Main model trained; test-set Dice/HD95 + volume agreement committed in `results/`
- [ ] Baseline comparison + at least one ablation reported
- [ ] Weights + model card on HF Hub
- [ ] Public app live: upload 4 sequences → overlay viewer, volumes table, NIfTI + report downloads
- [ ] README and report complete, with numbers traceable to `results/`
- [ ] Disclaimer shown in the app, README and model card
