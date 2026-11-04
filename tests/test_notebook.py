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
