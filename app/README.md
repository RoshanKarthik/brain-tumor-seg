---
title: Brain Tumor Segmentation
emoji: 🧠
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.31.0
python_version: "3.10"
app_file: app.py
pinned: false
short_description: Brain tumor segmentation and volumetry from BraTS-style MRI
models:
  - {{MODEL_REPO}}
tags:
  - medical-imaging
  - image-segmentation
  - brain-tumor
  - monai
---

# Brain tumor segmentation and volumetry (BraTS MRI)

**Research and education only. Not a medical device; not for clinical use.**

Upload the four co-registered, skull-stripped MRI sequences of one patient (FLAIR, T1,
T1ce, T2) as NIfTI files. A 3D SegResNet trained on BraTS 2021 segments necrotic core
(NCR), edema (ED) and enhancing tumor (ET) and reports their volumes in cm³ (NCR, ED,
ET, TC, WT). You can browse the overlay in three planes and download the segmentation
(`segmentation.nii.gz`, original image space) and a report (JSON / CSV).

- Model weights and model card: [{{MODEL_REPO}}](https://huggingface.co/{{MODEL_REPO}})
- Code: <https://github.com/RoshanKarthik/brain-tumor-seg>

MRI only (CT is not supported); trained on adult pre-operative gliomas. Example scans
are not included because of the BraTS data terms.

Hardware: this Space is meant to run on **ZeroGPU**, which is selected in the Space's
Settings (it is not a README option). Without a GPU it falls back to the CPU.
