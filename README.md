# Brain Tumor Segmentation & Volumetric Quantification

Segments brain tumors in multi-sequence MRI (FLAIR, T1, T1ce, T2) into necrotic core (NCR), edema (ED)
and enhancing tumor (ET), and reports per-region volumes (NCR, ED, ET, TC, WT in cm³).
Trained on BraTS 2021 with MONAI. MRI only.

> **Disclaimer:** research and education only. Not a medical device and not for clinical use.

Status: early development. See [PROJECT_PLAN.md](PROJECT_PLAN.md). Results: TBD.

## Local setup (CPU)

Python 3.10 (a recent 3.10.x patch release: 3.10.0 has a dataclass bug that breaks pytest 9).

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; on Linux/macOS: source .venv/bin/activate
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
ruff check .
pytest
```
