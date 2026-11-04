"""The Kaggle notebook is valid JSON and every code cell is valid Python (IPython magics aside)."""

import json
import re
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_train.ipynb"


def test_notebook_cells_compile():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 5
    for i, cell in enumerate(code_cells):
        assert not cell["outputs"], "commit the notebook without outputs"
        # shell (!) and magic (%) lines become `pass`, keeping indentation
        src = re.sub(r"^(\s*)[!%].*$", r"\1pass", "".join(cell["source"]), flags=re.M)
        compile(src, f"{NOTEBOOK.name}[code cell {i}]", "exec")


def test_notebook_never_contains_tokens():
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert not re.search(r"hf_[A-Za-z0-9]{20,}", text)
    assert "HF_TOKEN" in text and "UserSecretsClient" in text


def test_kaggle_push_builds_kernel_folder(tmp_path):
    import importlib.util

    script = NOTEBOOK.parents[1] / "scripts" / "kaggle_push.py"
    spec = importlib.util.spec_from_file_location("kaggle_push", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    folder = module.main(["--user", "someone", "--mode", "full", "--resume",
                          "--config", "configs/unet3d.yaml", "--out", str(tmp_path)])
    meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "someone/brats-unet3d-train"
    assert meta["enable_gpu"] and meta["enable_internet"] and meta["is_private"]
    assert meta["dataset_sources"] == ["dschettler8845/brats-2021-task1"]
    assert meta["kernel_sources"] == ["someone/brats-unet3d-train"]

    nb = json.loads((folder / "kaggle_train.ipynb").read_text(encoding="utf-8"))
    settings = "".join(nb["cells"][1]["source"])
    assert 'CONFIG = "configs/unet3d.yaml"' in settings
    assert "RUN_SMOKE_TEST = False" in settings and "RUN_FULL_TRAINING = True" in settings
    assert "# the real run" in settings  # trailing comments survive

    smoke = module.main(["--user", "someone", "--out", str(tmp_path)])
    meta = json.loads((smoke / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "someone/brats-segresnet-smoke" and meta["kernel_sources"] == []
    nb = json.loads((smoke / "kaggle_train.ipynb").read_text(encoding="utf-8"))
    assert 'SMOKE_ARGS = "--epochs 2 --max-train-cases 20' in "".join(nb["cells"][1]["source"])

    timing = module.main(["--user", "someone", "--out", str(tmp_path), "--slug", "t",
                          "--smoke-args", "--epochs 3 --max-train-cases 100"])
    nb = json.loads((timing / "kaggle_train.ipynb").read_text(encoding="utf-8"))
    assert 'SMOKE_ARGS = "--epochs 3 --max-train-cases 100"' in "".join(nb["cells"][1]["source"])


def test_kaggle_push_eval_mode_attaches_training_output(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "kaggle_push", NOTEBOOK.parents[1] / "scripts" / "kaggle_push.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    folder = module.main(["--user", "someone", "--mode", "eval", "--out", str(tmp_path),
                          "--eval-args", "--split val"])
    meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "someone/brats-segresnet-eval"
    assert meta["kernel_sources"] == ["someone/brats-segresnet-train"]
    nb = json.loads((folder / "kaggle_train.ipynb").read_text(encoding="utf-8"))
    settings = "".join(nb["cells"][1]["source"])
    assert "RUN_EVALUATION = True" in settings and 'EVAL_ARGS = "--split val"' in settings
    assert "RUN_FULL_TRAINING = False" in settings and "RUN_SMOKE_TEST = False" in settings
